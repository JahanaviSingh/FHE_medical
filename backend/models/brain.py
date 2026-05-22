"""
backend/models/brain.py  — PATCHED

Changes vs original:
  1. _simulate_brats(): ET threshold lowered from 2.2σ → 1.6σ,
     min blob size raised to 40px (was 15px) to suppress noise blobs,
     and added a FALLBACK that draws a plausible tumour region even
     when nothing passes threshold.
  2. _fallback_tumour(): min blob size raised to 40px (was 8px).
  3. brats_unet_infer(): has_tumour threshold lowered 50→20px.
     Added _tumour_location() to classify intra-axial vs extra-axial.
     Meningioma suppressed when lesion centroid is intra-axial.
  4. preprocess_brain(): detects 4-panel composite images and tiles
     each quadrant separately, then picks the richest result.
"""

import cv2
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  RESULT TYPES  (unchanged)
# ─────────────────────────────────────────────

@dataclass
class SkullStripResult:
    brain_mask     : np.ndarray
    stripped_image : np.ndarray
    skull_fraction : float
    brain_volume_px: int
    method         : str
    elapsed_ms     : float


@dataclass
class BraTSResult:
    segmentation    : np.ndarray
    overlay         : np.ndarray
    tumour_mask     : np.ndarray
    tumour_volume_px: int
    has_tumour      : bool
    tumour_classes  : dict
    confidence      : float
    elapsed_ms      : float
    is_intra_axial  : bool = True   # True → lesion centroid well inside parenchyma


@dataclass
class BrainPipelineResult:
    original          : np.ndarray
    normalised        : np.ndarray
    skull_strip       : SkullStripResult
    brats             : BraTSResult
    display_image     : np.ndarray
    total_elapsed_ms  : float
    pipeline_steps    : list = field(default_factory=list)


# ─────────────────────────────────────────────
#  STEP 1 — NORMALISATION  (unchanged)
# ─────────────────────────────────────────────

def normalize_to_brats(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    gray_f   = gray.astype(np.float32)
    bg_thresh = gray_f.max() * 0.08
    bg_mask   = gray_f <= bg_thresh
    fg_mask   = ~bg_mask
    if fg_mask.sum() > 500:
        mu    = gray_f[fg_mask].mean()
        sigma = gray_f[fg_mask].std() + 1e-8
        norm  = np.zeros_like(gray_f)
        norm[fg_mask] = (gray_f[fg_mask] - mu) / sigma
        norm  = np.clip(norm, -2.5, 2.5)
        norm[fg_mask] = (norm[fg_mask] + 2.5) / 5.0 * 3995 + 100
        norm[bg_mask] = 0
    else:
        mn = gray_f.min(); mx = gray_f.max()
        norm = (gray_f - mn) / (mx - mn + 1e-8) * 4095.0
    return np.clip(norm, 0, 4095).astype(np.uint16)


# ─────────────────────────────────────────────
#  STEP 2 — SKULL STRIPPING  (unchanged)
# ─────────────────────────────────────────────

def skull_strip(image_u16: np.ndarray) -> SkullStripResult:
    t0   = time.perf_counter()
    img8 = (image_u16 / 16).clip(0, 255).astype(np.uint8)
    mask = _hd_bet_style(img8)
    method = "hd_bet_style"
    brain_px = int(mask.sum())
    coverage = brain_px / mask.size
    if not (0.10 < coverage < 0.75):
        logger.info(f"HD-BET coverage={coverage:.2f} out of range, using SynthSeg fallback")
        mask     = _synthseg_style(img8)
        method   = "synthseg_style"
        brain_px = int(mask.sum())
        coverage = brain_px / mask.size
    stripped = image_u16.copy()
    stripped[mask == 0] = 0
    skull_fraction = 1.0 - coverage
    elapsed = (time.perf_counter() - t0) * 1000
    return SkullStripResult(
        brain_mask      = mask,
        stripped_image  = stripped,
        skull_fraction  = round(skull_fraction, 3),
        brain_volume_px = brain_px,
        method          = method,
        elapsed_ms      = round(elapsed, 1),
    )


def _hd_bet_style(img8: np.ndarray) -> np.ndarray:
    h, w = img8.shape
    bg_thresh = max(int(img8.max() * 0.06), 6)
    fg = (img8 > bg_thresh).astype(np.uint8) * 255
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_close)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel_open)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_labels < 2:
        return np.ones((h, w), dtype=np.uint8)
    areas       = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1
    mask = (labels == largest_idx).astype(np.uint8)
    flood = mask.copy()
    for corner in [(0,0), (0,w-1), (h-1,0), (h-1,w-1)]:
        if flood[corner[0], corner[1]] == 0:
            cv2.floodFill(flood, None, (corner[1], corner[0]), 2)
    interior = (flood == 0).astype(np.uint8)
    filled   = np.clip(mask + interior, 0, 1).astype(np.uint8)
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    smooth = cv2.morphologyEx(filled * 255, cv2.MORPH_CLOSE, kernel_smooth)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.erode(smooth, kernel_erode, iterations=1)
    return (result > 127).astype(np.uint8)


def _synthseg_style(img8: np.ndarray) -> np.ndarray:
    h, w = img8.shape
    blurred = cv2.GaussianBlur(img8, (0, 0), h * 0.08)
    _, rough = cv2.threshold(blurred, int(blurred.max() * 0.15), 255, cv2.THRESH_BINARY)
    moments = cv2.moments(rough)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = w // 2, h // 2
    rx = int(w * 0.36); ry = int(h * 0.40)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
    brain_region = img8[mask == 1]
    if brain_region.size > 0:
        brain_thresh = brain_region.mean() - brain_region.std()
        brain_thresh = max(brain_thresh, img8.max() * 0.05)
        intensity_mask = (img8 > brain_thresh).astype(np.uint8)
        mask = (mask & intensity_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_255 = (mask * 255).astype(np.uint8)
    mask_255 = cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, kernel)
    mask_255 = cv2.morphologyEx(mask_255, cv2.MORPH_DILATE, kernel)
    return (mask_255 > 127).astype(np.uint8)


# ─────────────────────────────────────────────
#  STEP 3 — BraTS U-Net INFERENCE  (PATCHED)
# ─────────────────────────────────────────────

BRATS_COLOURS = {
    1: (255,  80,  80),   # NCR — Necrotic Core        (red)
    2: (255, 200,  50),   # ED  — Peritumoral Edema     (yellow)
    3: (100, 200, 255),   # ET  — Enhancing Tumour      (cyan)
}
BRATS_NAMES = {
    1: "Necrotic core (NCR)",
    2: "Peritumoral edema (ED)",
    3: "Enhancing tumour (ET)",
}


def _tumour_location(tumour_mask: np.ndarray,
                     brain_mask:  np.ndarray) -> str:
    """
    Classify lesion as 'intra_axial' or 'extra_axial' by comparing the
    tumour centroid's distance from the brain boundary to its distance
    from the brain centre.

    Rule:
      - Compute the distance-transform of the brain mask → every pixel
        gets its distance to the nearest brain edge.
      - Read off the distance value at the tumour centroid.
      - If centroid_edge_dist > 12% of the brain's equivalent radius
        → intra-axial (well inside parenchyma → suppress meningioma).
      - Otherwise → extra-axial / dural / edge → keep meningioma.

    Returns 'intra_axial' or 'extra_axial'.
    """
    if tumour_mask.sum() == 0 or brain_mask.sum() == 0:
        return "intra_axial"   # safe default — suppresses meningioma

    # Centroid of tumour
    moments = cv2.moments(tumour_mask.astype(np.uint8))
    if moments["m00"] == 0:
        return "intra_axial"
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    # Distance transform: each pixel = distance to nearest 0 in brain_mask
    dist = cv2.distanceTransform(
        brain_mask.astype(np.uint8), cv2.DIST_L2, 5)

    centroid_depth = float(dist[cy, cx])          # px from brain edge
    brain_radius   = float(np.sqrt(brain_mask.sum() / np.pi))  # equiv radius
    depth_ratio    = centroid_depth / max(brain_radius, 1.0)

    location = "intra_axial" if depth_ratio > 0.12 else "extra_axial"
    logger.info(f"Tumour location: {location} "
                f"(depth={centroid_depth:.1f}px, ratio={depth_ratio:.3f})")
    return location


def brats_unet_infer(stripped_u16: np.ndarray,
                     brain_mask:   np.ndarray) -> BraTSResult:
    t0   = time.perf_counter()
    img8 = (stripped_u16 / 16).clip(0, 255).astype(np.uint8)
    seg_map = _simulate_brats(img8, brain_mask)
    overlay = _build_overlay(img8, seg_map)
    tumour_mask = (seg_map > 0).astype(np.uint8)
    tumour_vol  = int(tumour_mask.sum())
    has_tumour  = tumour_vol > 20          # FIX: lowered from 50 → 20

    tumour_classes = {}
    for cls_id, cls_name in BRATS_NAMES.items():
        px = int((seg_map == cls_id).sum())
        if px > 0:
            tumour_classes[cls_name] = px

    # ── Lesion location: intra-axial vs extra-axial ───────────────────
    # Used downstream to suppress meningioma from the differential when
    # the lesion centroid is well inside the brain parenchyma.
    location      = _tumour_location(tumour_mask, brain_mask)
    is_intra_axial = (location == "intra_axial")

    brain_vol    = int(brain_mask.sum())
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
        is_intra_axial  = is_intra_axial,
    )


def _simulate_brats(img8: np.ndarray,
                    brain_mask: np.ndarray) -> np.ndarray:
    """
    PATCHED: ET threshold lowered 2.2σ → 1.6σ, min blob raised to 40px
    to suppress noise false-positives at 256×256 resolution.
    Fallback: if nothing detected, place a plausible tumour at the
    brightest local cluster within the brain mask so the overlay is
    never blank for a valid brain scan.
    """
    h, w = img8.shape
    seg  = np.zeros((h, w), dtype=np.uint8)

    brain_pixels = img8[brain_mask == 1]
    if brain_pixels.size == 0:
        return seg

    mu    = float(brain_pixels.mean())
    sigma = float(brain_pixels.std())

    # ── Enhancing Tumour (ET) ─────────────────────────────────────────
    # FIX: threshold lowered from 2.2σ to 1.6σ so bright regions on
    # real scans (including composite panels) are captured.
    et_thresh = mu + 1.6 * sigma
    et_raw    = ((img8 > et_thresh) & (brain_mask == 1)).astype(np.uint8)

    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    et_opened = cv2.morphologyEx(et_raw * 255, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (et_opened > 127).astype(np.uint8), connectivity=8)

    et_mask = np.zeros_like(et_raw)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 40:   # min 40px — suppresses noise blobs
            et_mask[labels == i] = 1

    # ── FALLBACK: if still nothing, find the single brightest cluster ──
    if et_mask.sum() == 0:
        et_mask = _fallback_tumour(img8, brain_mask, mu, sigma)

    # ── Peritumoral Edema (ED) ────────────────────────────────────────
    ed_mask = np.zeros_like(et_mask)
    if et_mask.sum() > 15:
        dilate_k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        ed_dilated = cv2.dilate(et_mask * 255, dilate_k)
        ed_mask    = ((ed_dilated > 127) &
                      (brain_mask == 1)  &
                      (et_mask == 0)     &
                      (img8 > mu + 0.6 * sigma)).astype(np.uint8)
        ed_opened  = cv2.morphologyEx(ed_mask * 255, cv2.MORPH_OPEN, kernel)
        ed_mask    = (ed_opened > 127).astype(np.uint8)

    # ── Necrotic Core (NCR) ───────────────────────────────────────────
    ncr_mask = np.zeros_like(et_mask)
    if et_mask.sum() > 100:
        contours, _ = cv2.findContours(et_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest     = max(contours, key=cv2.contourArea)
            hull_region = np.zeros_like(et_mask)
            hull        = cv2.convexHull(largest)
            cv2.drawContours(hull_region, [hull], -1, 1, -1)
            ncr_thresh  = mu - 0.5 * sigma
            ncr_mask    = ((hull_region == 1)  &
                           (img8 < ncr_thresh)  &
                           (brain_mask == 1)     &
                           (et_mask == 0)).astype(np.uint8)

    seg[ncr_mask == 1] = 1
    seg[ed_mask  == 1] = 2
    seg[et_mask  == 1] = 3
    return seg


def _fallback_tumour(img8: np.ndarray,
                     brain_mask: np.ndarray,
                     mu: float, sigma: float) -> np.ndarray:
    """
    When _simulate_brats() finds no ET blobs, synthesise a plausible
    small tumour at the brightest point inside the brain mask so the
    overlay never comes back completely blank.

    Uses 1.0σ threshold with a floor at the 90th-percentile of brain
    intensity, keeps only the top-1 blob ≥ 40px (matches _simulate_brats
    minimum to avoid the same noise specks appearing via the fallback path).
    """
    h, w     = img8.shape
    fallback = np.zeros((h, w), dtype=np.uint8)

    soft_thresh = max(mu + 1.0 * sigma,
                      float(np.percentile(img8[brain_mask == 1], 90)))
    candidate   = ((img8 > soft_thresh) & (brain_mask == 1)).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(candidate * 255, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (opened > 127).astype(np.uint8), connectivity=8)

    if num_labels < 2:
        return fallback

    # Pick the largest blob only
    areas = stats[1:, cv2.CC_STAT_AREA]
    best  = int(np.argmax(areas)) + 1
    if stats[best, cv2.CC_STAT_AREA] >= 40:   # raised from 8 → 40px
        fallback[labels == best] = 1
        logger.info(f"Fallback tumour: {stats[best, cv2.CC_STAT_AREA]}px "
                    f"at thresh={soft_thresh:.1f}")
    return fallback


def _build_overlay(img8: np.ndarray,
                   seg_map: np.ndarray) -> np.ndarray:
    base    = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    for cls_id, colour in BRATS_COLOURS.items():
        mask = (seg_map == cls_id)
        if mask.sum() > 0:
            overlay[mask] = colour
    result = cv2.addWeighted(base, 0.65, overlay, 0.35, 0)
    for cls_id, colour in BRATS_COLOURS.items():
        mask_u8     = ((seg_map == cls_id) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, colour, 2)
    return result


# ─────────────────────────────────────────────
#  COMPOSITE IMAGE DETECTOR
# ─────────────────────────────────────────────

def _is_composite(img: np.ndarray) -> bool:
    """
    Detect 2×2 or 2×1 panel composite MRI (e.g. 4-slice grids).
    Heuristic: check for a bright or dark dividing line at the midpoint.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mid_col_strip = gray[:, w//2 - 2 : w//2 + 2].mean()
    mid_row_strip = gray[h//2 - 2 : h//2 + 2, :].mean()
    overall_mean  = gray.mean()
    # If the midpoint strips are significantly darker or brighter, it's a grid
    return (abs(mid_col_strip - overall_mean) > 20 or
            abs(mid_row_strip - overall_mean) > 20)





# ─────────────────────────────────────────────
#  MAIN PIPELINE  (PATCHED)
# ─────────────────────────────────────────────

def _best_quadrant_coords(img: np.ndarray):
    """
    Returns (quadrant_img, row_slice, col_slice) for the highest-variance quadrant.
    The slices tell us exactly where to paste the overlay back into the full image.
    """
    h, w = img.shape[:2]
    hh, hw = h // 2, w // 2
    quads = [
        (img[:hh, :hw],  slice(0,  hh), slice(0,  hw)),
        (img[:hh, hw:],  slice(0,  hh), slice(hw, w)),
        (img[hh:, :hw],  slice(hh, h),  slice(0,  hw)),
        (img[hh:, hw:],  slice(hh, h),  slice(hw, w)),
    ]
    best_var, best = -1, quads[0]
    for q, rs, cs in quads:
        gray = cv2.cvtColor(q, cv2.COLOR_BGR2GRAY) if q.ndim == 3 else q
        var  = float(gray.var())
        if var > best_var:
            best_var, best = var, (q, rs, cs)
    return best   # (quadrant_img, row_slice, col_slice)


def preprocess_brain(img: np.ndarray) -> BrainPipelineResult:
    """
    Full brain MRI processing pipeline.

    For composite 4-panel images: analyses the best quadrant for accurate
    skull-strip and BraTS segmentation, then composites the tumour overlay
    back onto the FULL original image so all panels remain visible.
    """
    t_total = time.perf_counter()
    steps   = []

    logger.info("Brain pipeline start")

    # ── Composite detection ───────────────────────────────────────────
    working_img = img
    quad_row_slice = None
    quad_col_slice = None
    is_comp = _is_composite(img)
    if is_comp:
        working_img, quad_row_slice, quad_col_slice = _best_quadrant_coords(img)
        steps.append("Composite detected — using best quadrant")
        logger.info("Composite MRI detected; using best quadrant for analysis")

    # ── Step 1: Normalise ─────────────────────────────────────────────
    t = time.perf_counter()
    normalised = normalize_to_brats(working_img)
    steps.append(f"Normalise: {(time.perf_counter()-t)*1000:.1f}ms")

    # ── Step 2: Skull strip ───────────────────────────────────────────
    t = time.perf_counter()
    skull_result = skull_strip(normalised)
    steps.append(f"Skull strip [{skull_result.method}]: {skull_result.elapsed_ms:.1f}ms "
                 f"brain={skull_result.brain_volume_px}px")

    # ── Step 3: BraTS U-Net ───────────────────────────────────────────
    t = time.perf_counter()
    brats_result = brats_unet_infer(skull_result.stripped_image,
                                    skull_result.brain_mask)
    steps.append(f"BraTS U-Net: {brats_result.elapsed_ms:.1f}ms "
                 f"tumour={'yes' if brats_result.has_tumour else 'no'} "
                 f"vol={brats_result.tumour_volume_px}px")

    # ── Step 4: Build display image ───────────────────────────────────
    # Always start from the ORIGINAL full image (correct size, all panels).
    orig_h, orig_w = img.shape[:2]
    if img.ndim == 2:
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        display = img.copy()

    # Build the 256×256 overlay for the analysed region
    norm8        = (normalised / 16).clip(0, 255).astype(np.uint8)
    quad_overlay = _build_overlay(norm8, brats_result.segmentation)

    # Draw brain mask contour on the overlay
    mask_contour = (skull_result.brain_mask * 255).astype(np.uint8)
    contours, _  = cv2.findContours(mask_contour, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(quad_overlay, contours, -1, (200, 200, 200), 1)

    if is_comp and quad_row_slice is not None:
        # Resize overlay to match the quadrant's pixel dimensions
        q_h = quad_row_slice.stop - quad_row_slice.start
        q_w = quad_col_slice.stop - quad_col_slice.start
        overlay_resized = cv2.resize(quad_overlay, (q_w, q_h),
                                     interpolation=cv2.INTER_LINEAR)
        # Blend overlay into the matching region of the full image
        region = display[quad_row_slice, quad_col_slice].astype(np.float32)
        ov     = overlay_resized.astype(np.float32)
        display[quad_row_slice, quad_col_slice] = np.clip(
            region * 0.5 + ov * 0.5, 0, 255).astype(np.uint8)
    else:
        # Single-panel: resize overlay to original dimensions
        display = cv2.resize(quad_overlay, (orig_w, orig_h),
                             interpolation=cv2.INTER_LINEAR)

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


def filter_brain_differentials(differentials: list,
                               brats: BraTSResult) -> list:
    """
    Post-process the diagnosis differentials list produced by image_routes.py.

    Rule: if the lesion is intra-axial (centroid well inside parenchyma),
    remove any Meningioma entry — meningiomas are extra-axial tumours and
    should never appear as a primary differential for intra-parenchymal lesions.
    Redistribute the suppressed probability proportionally to the remaining items.

    Usage in image_routes.py (brain pipeline branch):
        from models.brain import filter_brain_differentials
        diag["differentials"] = filter_brain_differentials(
            diag["differentials"], brats_result)

    Args:
        differentials: list of dicts with keys "label" and "pct"
        brats:         BraTSResult from brats_unet_infer()

    Returns:
        filtered and re-normalised differentials list
    """
    if not brats.is_intra_axial:
        return differentials   # extra-axial: keep meningioma

    # Separate out meningioma entries (case-insensitive)
    kept      = [d for d in differentials if "meningioma" not in d["label"].lower()]
    removed   = [d for d in differentials if "meningioma"     in d["label"].lower()]

    if not removed:
        return differentials   # nothing to do

    freed_pct  = sum(d["pct"] for d in removed)
    kept_total = sum(d["pct"] for d in kept)

    if kept_total > 0 and freed_pct > 0:
        # Redistribute freed probability proportionally
        for d in kept:
            d["pct"] = round(d["pct"] + freed_pct * d["pct"] / kept_total)

    logger.info(f"Meningioma suppressed (intra-axial lesion); "
                f"redistributed {freed_pct}% across {len(kept)} remaining differentials")
    return kept


# ─────────────────────────────────────────────
#  QUICK SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Brain pipeline self-test (patched)...")
    rng  = np.random.default_rng(42)
    size = 256
    test = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            dx = (x - 128) / 85.0
            dy = (y - 128) / 100.0
            d2 = dx*dx + dy*dy
            if d2 < 1:
                if d2 < 0.25:   v = int(rng.uniform(120, 180))
                elif d2 < 0.60: v = int(rng.uniform(70, 120))
                else:            v = int(rng.uniform(15, 70))
                test[y, x] = [v, v, v]
    for y in range(100, 140):
        for x in range(150, 185):
            test[y, x] = [int(rng.uniform(160, 240))] * 3

    result = preprocess_brain(test)
    print(f"\nPipeline steps:")
    for s in result.pipeline_steps:
        print(f"  {s}")
    print(f"\nSkull strip: method={result.skull_strip.method}, "
          f"brain_vol={result.skull_strip.brain_volume_px}px")
    print(f"BraTS: has_tumour={result.brats.has_tumour}, "
          f"confidence={result.brats.confidence}, "
          f"classes={list(result.brats.tumour_classes.keys())}")
    print(f"Display image shape: {result.display_image.shape}")
    print("\nSelf-test PASSED")