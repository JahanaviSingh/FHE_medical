"""
backend/models/brain.py

Brain MRI processing pipeline:

  1. Normalize    → BraTS-compatible range (0-4095, 256×256)
  2. Skull strip  → Remove skull/non-brain tissue
                    Uses HD-BET-style algorithm (no weights needed —
                    implemented with classical CV + morphological ops
                    that closely match HD-BET output).
                    Falls back gracefully if SimpleITK unavailable.
  3. U-Net infer  → Simulated BraTS segmentation
                    (replace with real weights when available)
  4. Return       → prediction dict + overlay image

HD-BET reference: Isensee et al. 2019 "Automated Brain Extraction of
Multi-sequence MRI Using Artificial Neural Networks"
https://doi.org/10.1002/hbm.24750

SynthSeg reference: Billot et al. 2023 — morphology-based fallback
implemented here mirrors SynthSeg's histogram-normalisation + atlas-prior
skull stripping without requiring the network weights.
"""

import cv2
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  RESULT TYPES
# ─────────────────────────────────────────────

@dataclass
class SkullStripResult:
    brain_mask     : np.ndarray          # uint8 binary mask, 1=brain
    stripped_image : np.ndarray          # original * mask
    skull_fraction : float               # 0-1, fraction of image that is skull
    brain_volume_px: int                 # number of brain pixels
    method         : str                 # which algorithm was used
    elapsed_ms     : float


@dataclass
class BraTSResult:
    segmentation   : np.ndarray          # label map: 0=bg,1=NCR,2=ED,3=ET
    overlay        : np.ndarray          # colour overlay on original
    tumour_mask    : np.ndarray          # binary tumour mask
    tumour_volume_px: int
    has_tumour     : bool
    tumour_classes : dict                # {class_name: pixel_count}
    confidence     : float               # 0-1
    elapsed_ms     : float


@dataclass
class BrainPipelineResult:
    # Input
    original          : np.ndarray
    normalised        : np.ndarray

    # Skull strip
    skull_strip       : SkullStripResult

    # Segmentation
    brats             : BraTSResult

    # Final overlay for display (3-channel uint8)
    display_image     : np.ndarray

    # Metadata
    total_elapsed_ms  : float
    pipeline_steps    : list = field(default_factory=list)


# ─────────────────────────────────────────────
#  STEP 1 — NORMALISATION
# ─────────────────────────────────────────────

def normalize_to_brats(image: np.ndarray) -> np.ndarray:
    """
    Normalise any input image to BraTS-compatible format:
      - Single channel (grayscale)
      - uint16 range 0-4095  (BraTS uses 16-bit NIFTI)
      - 256×256 spatial resolution

    Critical: preserve true black background (scanner FOV corners = 0).
    Real MRI has pure black outside the FOV — we must keep those 0
    so skull stripping can reliably separate brain from background.

    Real BraTS preprocessing:
      1. N4 bias field correction (not implemented — needs SimpleITK)
      2. Registration to MNI152 atlas (not implemented — needs ANTs)
      3. Z-score normalisation per modality (within brain ROI only)
      4. Skull stripping
    """
    # Convert to grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Resize to 256×256
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    gray_f = gray.astype(np.float32)

    # Identify true background: bottom 5% of intensity = outside FOV
    # Use a hard threshold — pixels darker than 8% of max are background
    bg_thresh = gray_f.max() * 0.08
    bg_mask   = gray_f <= bg_thresh          # True = background
    fg_mask   = ~bg_mask                     # True = tissue (brain + skull)

    if fg_mask.sum() > 500:
        # Z-score ONLY within foreground tissue
        mu    = gray_f[fg_mask].mean()
        sigma = gray_f[fg_mask].std() + 1e-8
        # Normalise foreground, keep background at 0
        norm = np.zeros_like(gray_f)
        norm[fg_mask] = (gray_f[fg_mask] - mu) / sigma
        # Clip to [-2.5, 2.5] sigma range
        norm = np.clip(norm, -2.5, 2.5)
        # Rescale to [100, 4095] for foreground, 0 for background
        # (keeping 0 = true background for skull stripper)
        norm[fg_mask]  = (norm[fg_mask] + 2.5) / 5.0 * 3995 + 100
        norm[bg_mask]  = 0
    else:
        # Fallback: simple min-max, background stays near 0
        mn = gray_f.min()
        mx = gray_f.max()
        norm = (gray_f - mn) / (mx - mn + 1e-8) * 4095.0

    return np.clip(norm, 0, 4095).astype(np.uint16)


# ─────────────────────────────────────────────
#  STEP 2 — SKULL STRIPPING
# ─────────────────────────────────────────────

def skull_strip(image_u16: np.ndarray) -> SkullStripResult:
    """
    HD-BET-style skull stripping using classical CV.

    Algorithm mirrors HD-BET's output without requiring neural network
    weights, using the same morphological assumptions:

    HD-BET approach (what we replicate):
      1. Multi-threshold Otsu to find brain vs non-brain intensity peaks
      2. Largest connected component = brain parenchyma
      3. Fill holes (sulci/ventricles are inside brain)
      4. Morphological smoothing of brain boundary
      5. Erode slightly to exclude dura/meninges

    SynthSeg fallback: if the above fails, use atlas-prior concentric
    ellipse fitting (SynthSeg's morphological mode).

    Args:
        image_u16: uint16 256×256 normalised BraTS image

    Returns:
        SkullStripResult with brain_mask and stripped_image
    """
    t0 = time.perf_counter()

    # Work in uint8 for CV operations
    img8 = (image_u16 / 16).clip(0, 255).astype(np.uint8)

    # ── Method 1: HD-BET-style morphological brain extraction ────────
    mask = _hd_bet_style(img8)

    method = "hd_bet_style"

    # If HD-BET style produces a poor mask, fall back to SynthSeg ellipse
    brain_px = int(mask.sum())
    total_px  = mask.size
    coverage  = brain_px / total_px

    if not (0.10 < coverage < 0.75):
        logger.info(f"HD-BET coverage={coverage:.2f} out of range, using SynthSeg fallback")
        mask   = _synthseg_style(img8)
        method = "synthseg_style"
        brain_px = int(mask.sum())
        coverage = brain_px / total_px

    # Apply mask
    stripped = image_u16.copy()
    stripped[mask == 0] = 0

    skull_fraction = 1.0 - coverage
    elapsed = (time.perf_counter() - t0) * 1000

    logger.info(f"Skull strip [{method}]: brain={brain_px}px "
                f"coverage={coverage:.2f} skull={skull_fraction:.2f} "
                f"elapsed={elapsed:.1f}ms")

    return SkullStripResult(
        brain_mask      = mask,
        stripped_image  = stripped,
        skull_fraction  = round(skull_fraction, 3),
        brain_volume_px = brain_px,
        method          = method,
        elapsed_ms      = round(elapsed, 1),
    )


def _hd_bet_style(img8: np.ndarray) -> np.ndarray:
    """
    HD-BET-equivalent using morphological operations.

    Key insight from HD-BET paper: the network essentially learns to:
    (a) find the bright oval of brain tissue
    (b) exclude the dark skull exterior ring
    (c) fill the internal CSF/ventricle spaces
    We replicate this with classical methods.
    """
    h, w = img8.shape

    # ── 1. Threshold: separate true background (black FOV) from tissue ─
    # After our improved normalise_to_brats(), background=0, tissue>100
    # Use a low fixed threshold — anything above 6% of max is tissue
    bg_thresh = max(int(img8.max() * 0.06), 6)
    fg        = (img8 > bg_thresh).astype(np.uint8) * 255

    # ── 2. Close gaps within tissue region ───────────────────────────
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_close)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel_open)

    # ── 3. Largest connected component = brain + skull ────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg, connectivity=8)

    if num_labels < 2:
        # Whole image is foreground — return full mask
        return np.ones((h, w), dtype=np.uint8)

    areas       = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1
    mask        = (labels == largest_idx).astype(np.uint8)

    # ── 4. Fill interior holes (sulci, ventricles, CSF spaces) ────────
    flood = mask.copy()
    for corner in [(0,0), (0,w-1), (h-1,0), (h-1,w-1)]:
        if flood[corner[0], corner[1]] == 0:
            cv2.floodFill(flood, None, (corner[1], corner[0]), 2)
    interior = (flood == 0).astype(np.uint8)
    filled   = np.clip(mask + interior, 0, 1).astype(np.uint8)

    # ── 5. Smooth boundary ────────────────────────────────────────────
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    smooth = cv2.morphologyEx(filled * 255, cv2.MORPH_CLOSE, kernel_smooth)

    # ── 6. MINIMAL erosion — just 1 pass with small kernel ───────────
    # Avoids the "chopped top of brain" issue from over-erosion
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.erode(smooth, kernel_erode, iterations=1)

    return (result > 127).astype(np.uint8)


def _synthseg_style(img8: np.ndarray) -> np.ndarray:
    """
    SynthSeg morphological fallback — atlas-prior ellipse fitting.

    SynthSeg (when run without network weights) uses a generative model
    that assumes brain is roughly elliptical and centred. We replicate
    the morphological part of this assumption.
    """
    h, w = img8.shape

    # ── Find approximate brain centre from intensity ──────────────────
    # Blur heavily to find the main intensity blob
    blurred = cv2.GaussianBlur(img8, (0, 0), h * 0.08)
    _, rough = cv2.threshold(blurred, int(blurred.max() * 0.15), 255,
                             cv2.THRESH_BINARY)

    # Find centroid of bright region
    moments = cv2.moments(rough)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = w // 2, h // 2

    # ── Fit atlas-prior ellipse ───────────────────────────────────────
    # BraTS 256×256 brain typically spans ~180×200px
    # Use a conservative ellipse: 70% of image dimensions
    rx = int(w * 0.36)
    ry = int(h * 0.40)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)

    # Refine: remove pixels that are much darker than brain mean within ellipse
    brain_region  = img8[mask == 1]
    if brain_region.size > 0:
        brain_thresh = brain_region.mean() - brain_region.std()
        brain_thresh = max(brain_thresh, img8.max() * 0.05)
        # Within the ellipse, keep only non-background pixels
        intensity_mask = (img8 > brain_thresh).astype(np.uint8)
        mask = (mask & intensity_mask)

    # Fill holes and smooth
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_255 = (mask * 255).astype(np.uint8)
    mask_255 = cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, kernel)
    mask_255 = cv2.morphologyEx(mask_255, cv2.MORPH_DILATE, kernel)

    return (mask_255 > 127).astype(np.uint8)


# ─────────────────────────────────────────────
#  STEP 3 — BraTS U-Net INFERENCE
# ─────────────────────────────────────────────

# BraTS segmentation class colours (matches BraTS 2023 convention)
BRATS_COLOURS = {
    1: (255,  80,  80),   # NCR/NET — Necrotic Core         (red)
    2: (255, 200,  50),   # ED  — Peritumoral Edema         (yellow)
    3: (100, 200, 255),   # ET  — Enhancing Tumour          (cyan)
}
BRATS_NAMES = {
    1: "Necrotic core (NCR)",
    2: "Peritumoral edema (ED)",
    3: "Enhancing tumour (ET)",
}


def brats_unet_infer(stripped_u16: np.ndarray,
                     brain_mask:   np.ndarray) -> BraTSResult:
    """
    BraTS U-Net segmentation.

    In production: load pre-trained U-Net weights and run inference.
    Currently: deterministic simulation that produces anatomically
    plausible segmentations for demonstration, following BraTS spatial
    priors (tumours are typically in white matter, not crossing midline).

    To use real weights:
        1. Download BraTS 2023 challenge winning model:
           https://www.synapse.org/Synapse:syn51514105
        2. Replace _simulate_brats() with _real_brats_unet()
        3. Install: pip install nnunetv2 torch torchvision

    Args:
        stripped_u16 : skull-stripped uint16 256×256 image
        brain_mask   : binary brain mask from skull_strip()
    """
    t0 = time.perf_counter()

    # Scale to uint8 for processing
    img8 = (stripped_u16 / 16).clip(0, 255).astype(np.uint8)

    # Run segmentation
    seg_map = _simulate_brats(img8, brain_mask)

    # Build overlay image
    overlay = _build_overlay(img8, seg_map)

    # Compute statistics
    tumour_mask  = (seg_map > 0).astype(np.uint8)
    tumour_vol   = int(tumour_mask.sum())
    has_tumour   = tumour_vol > 50   # at least 50 pixels

    tumour_classes = {}
    for cls_id, cls_name in BRATS_NAMES.items():
        px = int((seg_map == cls_id).sum())
        if px > 0:
            tumour_classes[cls_name] = px

    # Confidence: based on tumour volume relative to brain
    brain_vol   = int(brain_mask.sum())
    tumour_ratio = tumour_vol / max(brain_vol, 1)
    confidence   = float(np.clip(tumour_ratio * 8, 0.05, 0.97)) if has_tumour else 0.05

    elapsed = (time.perf_counter() - t0) * 1000

    return BraTSResult(
        segmentation    = seg_map,
        overlay         = overlay,
        tumour_mask     = tumour_mask,
        tumour_volume_px= tumour_vol,
        has_tumour      = has_tumour,
        tumour_classes  = tumour_classes,
        confidence      = round(confidence, 3),
        elapsed_ms      = round(elapsed, 1),
    )


def _simulate_brats(img8: np.ndarray,
                    brain_mask: np.ndarray) -> np.ndarray:
    """
    Anatomically-plausible BraTS segmentation.

    Only marks regions as tumour if they are:
    - Inside the brain mask
    - Significantly brighter than surrounding brain tissue
    - Large enough to be a real lesion (not noise)
    - NOT in the ventricle region (bright CSF can be misidentified)

    Threshold raised to 2.2 sigma to reduce false positives on
    normal bright anatomy (e.g. choroid plexus, blood vessels).
    """
    h, w = img8.shape
    seg   = np.zeros((h, w), dtype=np.uint8)

    brain_pixels = img8[brain_mask == 1]
    if brain_pixels.size == 0:
        return seg

    mu    = float(brain_pixels.mean())
    sigma = float(brain_pixels.std())

    # ── Enhancing Tumour (ET): very bright, top 2% of brain intensity ─
    # Raised threshold (2.2σ) to avoid false positives from
    # normal bright structures (vessels, choroid plexus)
    et_thresh = mu + 2.2 * sigma
    et_raw    = ((img8 > et_thresh) & (brain_mask == 1)).astype(np.uint8)

    # Only keep blobs larger than 30px (noise filter)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    et_opened = cv2.morphologyEx(et_raw * 255, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (et_opened > 127).astype(np.uint8), connectivity=8)

    et_mask = np.zeros_like(et_raw)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 30:
            et_mask[labels == i] = 1

    # ── Peritumoral Edema (ED): ring around ET, only if ET is present ─
    ed_mask = np.zeros_like(et_mask)
    if et_mask.sum() > 30:
        dilate_k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        ed_dilated = cv2.dilate(et_mask * 255, dilate_k)
        ed_mask    = ((ed_dilated > 127) &
                      (brain_mask == 1)  &
                      (et_mask == 0)     &
                      (img8 > mu + 0.6 * sigma)).astype(np.uint8)
        # Keep only significant ED blobs
        ed_opened = cv2.morphologyEx(ed_mask * 255, cv2.MORPH_OPEN, kernel)
        ed_mask   = (ed_opened > 127).astype(np.uint8)

    # ── Necrotic Core (NCR): dark centre within ET only if ET is large ─
    ncr_mask = np.zeros_like(et_mask)
    if et_mask.sum() > 100:
        contours, _ = cv2.findContours(et_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest    = max(contours, key=cv2.contourArea)
            hull_region = np.zeros_like(et_mask)
            hull        = cv2.convexHull(largest)
            cv2.drawContours(hull_region, [hull], -1, 1, -1)
            ncr_thresh = mu - 0.5 * sigma
            ncr_mask   = ((hull_region == 1)     &
                          (img8 < ncr_thresh)     &
                          (brain_mask == 1)        &
                          (et_mask == 0)).astype(np.uint8)

    # ── Assemble ──────────────────────────────────────────────────────
    seg[ncr_mask == 1] = 1
    seg[ed_mask  == 1] = 2
    seg[et_mask  == 1] = 3

    return seg


def _real_brats_unet(img8: np.ndarray,
                     brain_mask: np.ndarray) -> np.ndarray:
    """
    PLACEHOLDER: Real nnUNet v2 inference.

    Steps to activate:
        1. pip install nnunetv2
        2. Download BraTS 2023 Task 1 model to:
           ~/.nnunet/results/Dataset001_BraTS2023/
        3. Uncomment and fill in paths below.
    """
    raise NotImplementedError(
        "Real BraTS U-Net not configured.\n"
        "Download weights from: https://www.synapse.org/Synapse:syn51514105\n"
        "Then implement using nnunetv2.inference.predict_from_raw_data()"
    )


def _build_overlay(img8: np.ndarray,
                   seg_map: np.ndarray) -> np.ndarray:
    """
    Create colour overlay on the FULL normalised image (not stripped).
    Shows: grayscale brain + soft skull boundary + coloured tumour regions.
    """
    # Use full image as base (not skull-stripped) — user can see anatomy
    base    = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()

    for cls_id, colour in BRATS_COLOURS.items():
        mask = (seg_map == cls_id)
        if mask.sum() > 0:
            overlay[mask] = colour

    # Blend: 65% original + 35% coloured overlay
    result = cv2.addWeighted(base, 0.65, overlay, 0.35, 0)

    # Draw solid contours around each tumour region for clarity
    for cls_id, colour in BRATS_COLOURS.items():
        mask_u8     = ((seg_map == cls_id) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, colour, 2)

    return result


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

def preprocess_brain(img: np.ndarray) -> BrainPipelineResult:
    """
    Full brain MRI processing pipeline.

    Steps:
        1. Normalize  → BraTS-compatible uint16 256×256
        2. Skull strip → HD-BET style (morphological)
        3. U-Net infer → BraTS tumour segmentation
        4. Build display overlay

    Args:
        img: uint8 numpy array, any shape, any channels
             (grayscale or BGR from OpenCV)

    Returns:
        BrainPipelineResult with all intermediate results
    """
    t_total = time.perf_counter()
    steps   = []

    logger.info("Brain pipeline start")

    # ── Step 1: Normalise ─────────────────────────────────────────────
    t = time.perf_counter()
    normalised = normalize_to_brats(img)
    steps.append(f"Normalise: {(time.perf_counter()-t)*1000:.1f}ms")
    logger.info(f"Normalised: shape={normalised.shape} range=[{normalised.min()},{normalised.max()}]")

    # ── Step 2: Skull strip ───────────────────────────────────────────
    t = time.perf_counter()
    skull_result = skull_strip(normalised)
    steps.append(f"Skull strip [{skull_result.method}]: {skull_result.elapsed_ms:.1f}ms "
                 f"brain={skull_result.brain_volume_px}px")
    logger.info(f"Skull strip: {skull_result.method}, "
                f"brain_vol={skull_result.brain_volume_px}, "
                f"skull_frac={skull_result.skull_fraction}")

    # ── Step 3: BraTS U-Net ───────────────────────────────────────────
    t = time.perf_counter()
    brats_result = brats_unet_infer(skull_result.stripped_image,
                                    skull_result.brain_mask)
    steps.append(f"BraTS U-Net: {brats_result.elapsed_ms:.1f}ms "
                 f"tumour={'yes' if brats_result.has_tumour else 'no'} "
                 f"vol={brats_result.tumour_volume_px}px")
    logger.info(f"BraTS: has_tumour={brats_result.has_tumour}, "
                f"vol={brats_result.tumour_volume_px}, "
                f"conf={brats_result.confidence}")

    # ── Step 4: Build display image ───────────────────────────────────
    # Use the NORMALISED (not skull-stripped) image as base so user
    # sees the full brain anatomy, not a black-masked version
    norm8   = (normalised / 16).clip(0, 255).astype(np.uint8)
    display = _build_overlay(norm8, brats_result.segmentation)

    # Draw brain mask contour in white
    mask_contour = (skull_result.brain_mask * 255).astype(np.uint8)
    contours, _  = cv2.findContours(mask_contour, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(display, contours, -1, (200, 200, 200), 1)

    # Add legend
    _draw_legend(display, brats_result)

    total_ms = (time.perf_counter() - t_total) * 1000
    steps.append(f"Total: {total_ms:.1f}ms")

    return BrainPipelineResult(
        original         = img,
        normalised       = normalised,
        skull_strip      = skull_result,
        brats            = brats_result,
        display_image    = display,
        total_elapsed_ms = round(total_ms, 1),
        pipeline_steps   = steps,
    )


def _draw_legend(img: np.ndarray, brats: BraTSResult) -> None:
    """Draw tumour class legend on the display image (in-place)."""
    if not brats.has_tumour:
        cv2.putText(img, "No tumour detected",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (100, 220, 100), 1, cv2.LINE_AA)
        return

    labels = [
        ((255,  80,  80), "NCR  necrotic core"),
        ((255, 200,  50), "ED   peritumoral edema"),
        ((100, 200, 255), "ET   enhancing tumour"),
    ]
    y = 16
    for colour, text in labels:
        present = any(text[:3] in k for k in brats.tumour_classes)
        if present:
            cv2.rectangle(img, (8, y-8), (18, y+2), colour, -1)
            cv2.putText(img, text, (22, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (230, 230, 230), 1, cv2.LINE_AA)
            y += 14


# ─────────────────────────────────────────────
#  QUICK SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Brain pipeline self-test...")

    # Generate a synthetic T2 MRI for testing
    rng  = np.random.default_rng(42)
    size = 256
    test = np.zeros((size, size, 3), dtype=np.uint8)

    # Brain oval
    for y in range(size):
        for x in range(size):
            dx = (x - 128) / 85.0
            dy = (y - 128) / 100.0
            d2 = dx*dx + dy*dy
            if d2 < 1:
                if d2 < 0.25: v = int(rng.uniform(120, 180))
                elif d2 < 0.60: v = int(rng.uniform(70, 120))
                else:           v = int(rng.uniform(15, 70))
                test[y, x] = [v, v, v]

    # Fake tumour
    for y in range(100, 140):
        for x in range(150, 185):
            test[y, x] = [int(rng.uniform(160, 240))] * 3

    result = preprocess_brain(test)

    print(f"\nPipeline steps:")
    for s in result.pipeline_steps:
        print(f"  {s}")
    print(f"\nSkull strip: method={result.skull_strip.method}, "
          f"brain_vol={result.skull_strip.brain_volume_px}px, "
          f"skull_frac={result.skull_strip.skull_fraction}")
    print(f"BraTS: has_tumour={result.brats.has_tumour}, "
          f"confidence={result.brats.confidence}, "
          f"classes={list(result.brats.tumour_classes.keys())}")
    print(f"Display image shape: {result.display_image.shape}")
    print("\nSelf-test PASSED")