"""
backend/utils/validator.py

Fixed issues:
1. extract_features() is IDENTICAL to train_validator_multiclass.py
2. Thresholds relaxed: xray=0.35, mri=0.30, bone=0.05, ct=0.35
3. Float boundary bug fixed: >= instead of > for ml_prob check
4. _pixel_stats() channel order bug fixed (was swapping R/B)
5. Composite image handling (4-panel CT grids)
6. Blue tint allowed for MRI/CT (chroma up to 120)
7. Physics override for MRI/CT when physics all pass
8. Full debug logging of every feature + per-class probability
9. Score-based fallback for bone when ML uncertain
10. Grayscale images handled consistently
"""

import numpy as np
import cv2
import pickle
import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MODEL_PATH      = os.path.join(os.path.dirname(__file__), "scan_classifier.pkl")
MULTICLASS_PATH = os.path.join(os.path.dirname(__file__), "scan_classifier_multiclass.pkl")
MODALITY_TO_CLASS = {"xray": 1, "mri": 2, "bone": 3, "ct": 4}

# Relaxed thresholds — use >= comparison so 0.35 >= 0.35 PASSES
MODALITY_THRESHOLDS = {
    "xray": {
        "ml_prob": 0.35,          # was 0.40 — fixes prob=0.40 boundary
        "lung_required": True,
        "hf_min": 0.05, "hf_max": 99.0,
        "edge_max": 0.14,
        "label": "Chest X-ray",
        "lung_area": (0.25, 0.45),
    },
    "mri": {
        "ml_prob": 0.30,          # relaxed — MRI blue tint confuses old models
        "lung_required": False,
        "hf_min": 0.5, "hf_max": 99.0,
        "edge_max": 0.25,
        "label": "Brain MRI",
        "isotropy_min": 0.28,
    },
    "bone": {
        "ml_prob": 0.05,          # very relaxed — bone varies hugely
        "lung_required": False,
        "hf_min": 0.05, "hf_max": 99.0,
        "edge_max": 0.35,
        "label": "Bone X-ray",
        "edge_density_min": 0.02,
    },
    "ct": {
        "ml_prob": 0.35,          # relaxed — blue-tinted CTs common
        "lung_required": False,
        "hf_min": 0.05, "hf_max": 99.0,
        "edge_max": 0.25,
        "label": "CT scan",
        "trimodal": True,
    },
}

FEATURE_NAMES = [
    "chroma", "sat", "colour_frac", "edge_density", "lap_var",
    "int_range", "mean_int", "std_int", "hist_entropy", "hist_max",
    "green_dom", "red_dom", "hf_energy", "local_contrast", "blob_diff",
    "grad_mean", "cb_ratio", "dark_frac", "bright_frac", "peaks64", "std_norm",
]


# ─────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class CheckResult:
    label: str
    passed: bool
    value: str
    expected: str
    detail: str = ""

@dataclass
class ValidationReport:
    modality: str
    status: str
    score: float
    message: str
    hint: str
    checks: list = field(default_factory=list)
    pixel_stats: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
#  FEATURE EXTRACTOR
#  MUST be identical to train_validator_multiclass.py
# ─────────────────────────────────────────────

def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Extract 21 features. This function is the single source of truth —
    identical copy exists in train_validator_multiclass.py.
    Any change here must be mirrored there.
    """
    img = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # BGR order — index 2=R, 1=G, 0=B
    r = img[:,:,2].astype(float)
    g = img[:,:,1].astype(float)
    b = img[:,:,0].astype(float)
    gray = (r*0.299 + g*0.587 + b*0.114).astype(np.uint8)

    chroma = abs(r.mean()-g.mean()) + abs(g.mean()-b.mean()) + abs(r.mean()-b.mean())
    max_c = np.maximum(np.maximum(r,g),b)
    min_c = np.minimum(np.minimum(r,g),b)
    sat = (max_c - min_c) / (max_c + 1e-6)
    colour_frac = float(((max_c-min_c) > 30).mean())

    edges = cv2.Canny(gray, 40, 120)
    edge_density = float(edges.sum()) / (255.0 * 128 * 128)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())
    int_range = int(gray.max()) - int(gray.min())

    hist, _ = np.histogram(gray.flatten(), bins=32, range=(0,255))
    hn = hist.astype(float) / (hist.sum() + 1e-6)
    hist_entropy = float(-np.sum(hn * np.log(hn + 1e-10)))
    hist_max = float(hn.max())

    green_dom = float(((g > r+10) & (g > b+10)).mean())
    red_dom   = float(((r > g+15) & (r > b+15)).mean())

    gray_f = gray.astype(np.float32)
    dct = cv2.dct(gray_f)
    total_e = (dct**2).sum() + 1e-6
    h2, w2 = dct.shape
    hf_energy = float(((dct[h2//4:, w2//4:])**2).sum() / total_e)

    kernel = np.ones((8,8), np.float32) / 64
    local_contrast = float(cv2.filter2D(gray.astype(float), -1, kernel).std())
    blurred = cv2.GaussianBlur(gray, (15,15), 5)
    blob_diff = float(np.abs(gray.astype(float) - blurred.astype(float)).mean())

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mean = float(np.sqrt(gx**2 + gy**2).mean())

    h, w = gray.shape
    margin = min(h, w) // 4
    cy, cx = h // 2, w // 2
    center = gray[cy-margin:cy+margin, cx-margin:cx+margin]
    border = np.concatenate([
        gray[:margin,:].flatten(), gray[-margin:,:].flatten(),
        gray[:,:margin].flatten(), gray[:,-margin:].flatten()
    ])
    cb_ratio = float(center.mean()+1) / (float(border.mean())+1)

    dark_frac   = float((gray < 20).mean())
    bright_frac = float((gray > 180).mean())

    hist64, _ = np.histogram(gray.flatten(), bins=64, range=(0,255))
    h64 = hist64.astype(float) / (hist64.sum() + 1e-6)
    thr64 = h64.max() * 0.10
    peaks64 = 0
    in_p = False
    for v in h64:
        if v > thr64 and not in_p: peaks64 += 1; in_p = True
        elif v <= thr64: in_p = False

    return np.array([
        chroma, float(sat.mean()), colour_frac, edge_density, lap_var,
        int_range, float(gray.mean()), float(gray.std()),
        hist_entropy, hist_max, green_dom, red_dom,
        hf_energy, local_contrast, blob_diff,
        grad_mean, cb_ratio, dark_frac, bright_frac,
        float(peaks64), float(gray.std() / 128)
    ], dtype=np.float32)


# ─────────────────────────────────────────────
#  PHYSICS CHECKS
# ─────────────────────────────────────────────

def check_gradient_isotropy(gray):
    gx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    # Use lower threshold: mean only (not mean+std) so more pixels qualify
    threshold = mag.mean() * 0.5
    mask = mag > threshold
    if mask.sum() < 50:
        return 0.5   # not blank — return neutral, never 0
    hist, _ = np.histogram(angle[mask], bins=8, range=(-np.pi, np.pi))
    hn = hist.astype(float) / (hist.sum() + 1e-6)
    return float(np.clip(1.0 - hn.std() * 8.0, 0.0, 1.0))


def check_fft_spectrum(gray):
    fshift = np.fft.fftshift(np.fft.fft2(gray.astype(float)))
    mag = np.abs(fshift)
    h, w = mag.shape
    cy, cx = h//2, w//2
    def re(r1, r2):
        y, x = np.ogrid[:h, :w]
        d = np.sqrt((x-cx)**2 + (y-cy)**2)
        return float(mag[(d>=r1) & (d<r2)].sum())
    tot  = re(0, max(h,w)//2) + 1e-6
    low  = re(0,  10) / tot
    mid  = re(10, 30) / tot
    high = re(30, max(h,w)//2) / tot
    return {
        "fft_low":   round(low,  4),
        "fft_mid":   round(mid,  4),
        "fft_high":  round(high, 4),
        "fft_score": round(high / (low + 1e-6), 4),
    }


def check_histogram_shape(gray):
    hist, _ = np.histogram(gray.flatten(), bins=64, range=(0,255))
    hf = hist.astype(float) / (hist.sum() + 1e-6)
    sm = cv2.GaussianBlur(hf.reshape(1,-1).astype(np.float32), (1,9), 2.0).flatten()
    smoothness    = float(np.clip(1.0 - np.abs(hf-sm).mean() * 50, 0, 1))
    extreme_frac  = float(hf[:4].sum() + hf[-4:].sum())
    thr   = hf.max() * 0.12
    peaks = 0
    in_peak = False
    for v in hf:
        if v > thr and not in_peak: peaks += 1; in_peak = True
        elif v <= thr: in_peak = False
    return {
        "hist_smoothness": round(smoothness,   3),
        "extreme_frac":    round(extreme_frac, 3),
        "hist_peaks":      peaks,
    }


def check_lung_field(gray):
    h, w = gray.shape
    img_area = h * w
    best = {"lung_score":0.0,"lung_area_ratio":0.0,"lung_count":0,
            "convexity":0.0,"lateral_gap":0.0}
    for tm in [0.75, 0.85, 0.95]:
        _, dark = cv2.threshold(gray, int(gray.mean()*tm), 255, cv2.THRESH_BINARY_INV)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        dark = cv2.morphologyEx(cv2.morphologyEx(dark, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cands = [c for c in cnts if img_area*0.05 < cv2.contourArea(c) < img_area*0.45]
        if not cands: continue
        tot_area = sum(cv2.contourArea(c) for c in cands)
        ar = tot_area / img_area
        lg = max(cands, key=cv2.contourArea)
        hull = cv2.convexHull(lg)
        conv = cv2.contourArea(lg) / (cv2.contourArea(hull) + 1e-6)
        cnt  = len(cands)
        lat  = 0.0
        if cnt >= 2:
            cents = []
            for c in sorted(cands, key=cv2.contourArea, reverse=True)[:2]:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cents.append(M["m10"] / M["m00"])
            if len(cents) == 2:
                gf = abs(cents[1]-cents[0]) / w
                lat = 1.0 if 0.06 < gf < 0.45 else 0.0
        sc = (0.35*(1.0 if cnt>=2 else 0.0) + 0.35*lat +
              0.15*(1.0 if 0.18<ar<0.70 else 0.0) +
              0.15*(1.0 if conv>0.72 else 0.0))
        if sc > best["lung_score"]:
            best = {
                "lung_score":      round(float(sc),   3),
                "lung_area_ratio": round(float(ar),   3),
                "lung_count":      cnt,
                "convexity":       round(float(conv), 3),
                "lateral_gap":     round(float(lat),  3),
            }
    return best


# ─────────────────────────────────────────────
#  MAIN VALIDATOR
# ─────────────────────────────────────────────

class MedicalImageValidator:

    MODALITIES = ["xray", "mri", "bone", "ct"]

    def __init__(self):
        self.model          = None
        self.scaler         = None
        self.is_multiclass  = False
        self._load_model()

    def _load_model(self):
        for path, is_multi in [(MULTICLASS_PATH, True), (MODEL_PATH, False)]:
            if os.path.exists(path):
                try:
                    d = pickle.load(open(path, "rb"))
                    self.model         = d["clf"]
                    self.scaler        = d["scaler"]
                    self.is_multiclass = is_multi
                    mtype = "Multi-class" if is_multi else "Binary"
                    print(f"[Validator] {mtype} model loaded: {os.path.basename(path)}")
                    return
                except Exception as e:
                    print(f"[Validator] Could not load {path}: {e}")
        print("[Validator] No model found — physics checks only")

    def validate(self, image: np.ndarray, modality: str) -> ValidationReport:
        if modality not in self.MODALITIES:
            modality = "xray"
        cfg   = MODALITY_THRESHOLDS[modality]
        label = cfg["label"]

        # ── Pre-processing ────────────────────────────────────────────
        # 1. Crop 5% border (removes watermarks, logos, scale bars)
        h0, w0 = image.shape[:2]
        crop = int(min(h0, w0) * 0.05)
        if crop > 5:
            image = image[crop:h0-crop, crop:w0-crop]

        # 2. Composite image detection (e.g. 4-panel CT grid)
        #    Extract top-left panel to avoid grid line artefacts
        h0, w0 = image.shape[:2]
        aspect = w0 / max(h0, 1)
        if aspect > 1.8:
            n_cols = 4 if aspect > 3.0 else 2
            pw = w0 // n_cols
            image = image[:h0//2, :pw] if h0 > pw else image[:, :pw]
        elif aspect < 0.55:
            image = image[:h0//2, :]

        # 3. Ensure 3-channel
        img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── Compute checks ────────────────────────────────────────────
        stats  = self._pixel_stats(img)
        iso    = check_gradient_isotropy(gray)
        fft    = check_fft_spectrum(gray)
        hsh    = check_histogram_shape(gray)
        lung   = (check_lung_field(gray) if modality == "xray"
                  else {"lung_score":0.5,"lung_area_ratio":0.3,
                        "lung_count":2,"convexity":0.8,"lateral_gap":1.0})

        self._current_modality = modality
        ml     = self._ml_prob(image)
        ml_all = self._ml_all_probs(image)

        checks = self._build_checks(stats, iso, fft, hsh, lung, ml, modality, cfg, img)

        # ── Weighted score ────────────────────────────────────────────
        ml_s   = ml
        an_s   = self._anat(lung, modality)
        ph_s   = self._phys(stats, fft, cfg, modality)
        tx_s   = self._tex(iso, hsh, modality)
        final  = 0.4*ml_s + 0.3*an_s + 0.2*ph_s + 0.1*tx_s

        # ── Full debug logging ────────────────────────────────────────
        feats = extract_features(image)
        print(f"\n[Validator:{modality}] ── feature dump ──")
        for name, val in zip(FEATURE_NAMES, feats):
            print(f"  {name:20s} {val:.4f}")
        print(f"  isotropy             {iso:.4f}")
        # Bone-specific debug
        if modality == "bone":
            img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(f"  [bone] min={img_g.min()} max={img_g.max()} "
                  f"mean={img_g.mean():.1f} std={img_g.std():.1f}")
            print(f"  [bone] p92={np.percentile(img_g,92):.1f} "
                  f"p96={np.percentile(img_g,96):.1f} "
                  f"p98={np.percentile(img_g,98):.1f}")
            print(f"  [bone] top-8% coverage = "
                  f"{float((img_g>np.percentile(img_g,92)).mean()):.4f}")
            edges_bone = cv2.Canny(img_g,40,120)
            print(f"  [bone] edge count = {int(edges_bone.sum()/255)}")
        if ml_all:
            print(f"  per-class probs:     " +
                  ", ".join(f"{k}={v:.3f}" for k,v in ml_all.items()))
        print(f"[Validator:{modality}] ml={ml_s:.3f} anat={an_s:.3f} "
              f"phys={ph_s:.3f} tex={tx_s:.3f} FINAL={final:.3f}")
        for c in checks:
            print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.label}: {c.value} "
                  f"(expected {c.expected})")

        # ── Decision logic ────────────────────────────────────────────
        # Physics override: if every non-ML check passes and ML is not
        # confidently "non-medical", allow through as WARN.
        non_ml   = [c for c in checks if c.label != "ML classifier"]
        phys_ok  = all(c.passed for c in non_ml)

        # Bone-specific fallback: if physics clearly shows bone pattern,
        # bypass ML entirely. ML is unreliable for bone due to training data issues.
        img_gray_bone = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p92_bone      = float(np.percentile(img_gray_bone, 92))
        top8_bone     = float((img_gray_bone > p92_bone).mean())
        bone_physics_strong = (
            modality == "bone" and
            stats["range"] > 100 and
            top8_bone > 0.03 and
            stats["chroma_diff"] < 120 and
            ml >= 0.005
        )

        physics_override = (
            (modality in ("mri","ct","bone") and phys_ok and ml >= 0.01) or
            bone_physics_strong
        )

        # FIX: use >= for threshold comparison (not >) to avoid boundary rejection
        ml_passes = ml >= cfg["ml_prob"]

        if not ml_passes and not physics_override:
            st   = "fail"
            msg  = f"Not a {label} · ML rejected (prob={ml:.3f} < {cfg['ml_prob']})"
            hint = (f"Please upload a real {label.lower()}. "
                    "Photos, portraits, and documents are rejected.")
            print(f"[Validator] REJECT: ML veto, no physics override")

        elif not ml_passes and physics_override:
            st   = "warn"
            msg  = f"Valid {label} · physics confirmed (ML uncertain prob={ml:.3f})"
            hint = ("All physics checks pass. "
                    "Run python train_validator_multiclass.py to improve ML accuracy.")
            print(f"[Validator] WARN: physics override applied")

        elif final >= 0.70:
            st   = "pass"
            msg  = f"Valid {label} · score={final:.2f}"
            hint = "Medical scan confirmed · pipeline unlocked"
            print(f"[Validator] PASS: score={final:.2f}")

        elif final >= 0.50:
            st   = "warn"
            msg  = f"Low confidence · score={final:.2f}"
            hint = f"May not be a real {label.lower()} — results may be unreliable"
            print(f"[Validator] WARN: low score={final:.2f}")

        else:
            st   = "fail"
            msg  = f"Not a {label} · score={final:.2f}"
            hint = (f"Please upload a real {label.lower()}. "
                    "Photos, portraits, and documents are rejected.")
            print(f"[Validator] REJECT: low score={final:.2f}")

        print(f"[Validator] → status={st} modality={modality} label='{label}'")

        return ValidationReport(
            modality    = modality,
            status      = st,
            score       = round(final, 2),
            message     = msg,
            hint        = hint,
            checks      = checks,
            pixel_stats = {
                **stats, **fft, **hsh,
                "isotropy": round(iso,   3),
                "ml_prob":  round(ml,    3),
                **{f"lung_{k}": v for k,v in lung.items()},
                **({"all_probs": ml_all} if ml_all else {}),
            },
        )

    # ── Check builder ────────────────────────────────────────────────

    def _build_checks(self, stats, iso, fft, hsh, lung, ml, modality, cfg, img=None):
        C  = CheckResult
        ch = []

        # Chroma: MRI/CT clinical viewers render in blue (chroma up to 120)
        chroma_thr = 120 if modality in ("mri", "ct") else 15
        ch.append(C("Colour space",
                    stats["chroma_diff"] < chroma_thr,
                    f"chroma={stats['chroma_diff']:.1f}",
                    f"< {chroma_thr}"))

        # ML — FIX: use >= not > so boundary value passes
        ch.append(C("ML classifier",
                    ml >= cfg["ml_prob"],
                    f"medical prob={ml:.3f}",
                    f">= {cfg['ml_prob']}"))

        # Isotropy
        ch.append(C("Gradient isotropy",
                    iso > 0.25,
                    f"isotropy={iso:.3f}",
                    "> 0.25 (organic anatomy)"))

        # Document check — MRI/CT have black backgrounds so raise threshold
        extreme_thr = 0.75 if modality in ("mri", "ct") else 0.40
        ch.append(C("Not a document",
                    hsh["extreme_frac"] < extreme_thr,
                    f"extreme px={hsh['extreme_frac']:.3f}",
                    f"< {extreme_thr}"))

        # FFT
        fft_ok = cfg["hf_min"] <= fft["fft_score"] <= cfg["hf_max"]
        ch.append(C("Frequency spectrum",
                    fft_ok,
                    f"HF/LF={fft['fft_score']:.3f}",
                    f"{cfg['hf_min']} – {cfg['hf_max']}"))

        # Modality-specific anatomical check
        if modality == "xray":
            aok = (lung["lung_count"] >= 2 and
                   lung["lateral_gap"] > 0 and
                   lung["lung_area_ratio"] > 0.12)
            ch.append(C("Dual lung fields", aok,
                        f"count={lung['lung_count']} gap={lung['lateral_gap']:.2f} "
                        f"area={lung['lung_area_ratio']:.2f}",
                        "2 laterally separated dark ovals"))

        elif modality == "mri":
            ch.append(C("Brain oval",
                        stats["center_brighter"],
                        f"centre-border={stats['center_corner_diff']:.1f}",
                        "centre brighter than border"))

        elif modality == "bone":
            # Adaptive percentile-based bone signal — works regardless of exposure
            if img is not None:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                p92       = float(np.percentile(img_gray, 92))
                bone_signal = float((img_gray > p92).mean())
            else:
                # Fallback to stats bright_fraction if img not available
                bone_signal = stats["bright_fraction"]
            int_range = stats["range"]
            bone_ok   = (int_range > 100 and bone_signal > 0.03)
            ch.append(C("Bone density / range",
                        bone_ok,
                        f"top8%={bone_signal:.3f} range={int_range}",
                        "top-8% signal > 3% AND range > 100"))

        elif modality == "ct":
            ch.append(C("CT histogram",
                        hsh["hist_peaks"] >= 2,
                        f"peaks={hsh['hist_peaks']}",
                        ">= 2 peaks (air / tissue / bone)"))
        return ch

    # ── Weighted sub-scores ──────────────────────────────────────────

    def _anat(self, lung, modality) -> float:
        if modality != "xray":
            return 0.70
        return float(round(
            0.40 * (1.0 if lung["lung_count"] >= 2    else 0.0) +
            0.35 * (1.0 if lung["lateral_gap"] > 0    else 0.0) +
            0.15 * (1.0 if 0.12 < lung["lung_area_ratio"] < 0.75 else 0.0) +
            0.10 * (1.0 if lung.get("convexity", 0) > 0.70 else 0.0), 3))

    def _phys(self, stats, fft, cfg, modality="xray") -> float:
        chroma_thr = 120 if modality in ("mri", "ct") else 15
        g = 1.0 if stats["chroma_diff"] < chroma_thr else max(0.0, 1.0-stats["chroma_diff"]/chroma_thr)
        r = 1.0 if stats["range"] > 150 else stats["range"] / 150
        f = 1.0 if fft["fft_score"] > cfg["hf_min"] else fft["fft_score"] / (cfg["hf_min"] + 1e-6)
        return float(round((g + r + f) / 3.0, 3))

    def _tex(self, iso, hsh, modality="xray") -> float:
        i   = 1.0 if iso > 0.25 else iso / 0.25
        thr = 0.75 if modality in ("mri","ct") else 0.40
        h   = 1.0 if hsh["extreme_frac"] < thr else max(0.0, 1.0-hsh["extreme_frac"]/thr)
        return float(round((i + h) / 2.0, 3))

    # ── ML probability ───────────────────────────────────────────────

    def _ml_prob(self, image) -> float:
        if self.model is None:
            return 0.5
        try:
            feats   = extract_features(image).reshape(1, -1)
            feats_s = self.scaler.transform(feats)
            probs   = self.model.predict_proba(feats_s)[0]
            if self.is_multiclass:
                mod     = getattr(self, "_current_modality", "xray")
                cls_idx = MODALITY_TO_CLASS.get(mod, 1)
                non_med = probs[0]
                med_p   = probs[cls_idx]
                # Penalty only if model is VERY confident it's non-medical
                if non_med > 0.70:
                    return float(med_p * 0.30)
                return float(med_p)
            return float(probs[1])
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return 0.5

    def _ml_all_probs(self, image) -> dict:
        """Return all class probabilities for debug logging."""
        if self.model is None or not self.is_multiclass:
            return {}
        try:
            feats   = extract_features(image).reshape(1, -1)
            feats_s = self.scaler.transform(feats)
            probs   = self.model.predict_proba(feats_s)[0]
            names   = {0:"non_med",1:"xray",2:"mri",3:"bone",4:"ct"}
            return {names[i]: round(float(p), 3) for i,p in enumerate(probs)}
        except Exception:
            return {}

    # ── Pixel statistics (for physics checks) ───────────────────────

    def _pixel_stats(self, img: np.ndarray) -> dict:
        h, w = img.shape[:2]
        # FIX: correct BGR channel order
        b = img[:,:,0].astype(float)
        g = img[:,:,1].astype(float)
        r = img[:,:,2].astype(float)
        gray = (r*0.299 + g*0.587 + b*0.114).astype(np.uint8)

        chroma = float(abs(r.mean()-g.mean()) +
                       abs(g.mean()-b.mean()) +
                       abs(r.mean()-b.mean()))
        mn, mx = int(gray.min()), int(gray.max())

        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(edges.sum()) / (255 * h * w)

        max_c = np.maximum(np.maximum(r,g),b)
        min_c = np.minimum(np.minimum(r,g),b)
        sat   = (max_c - min_c) / (max_c + 1e-6)

        margin  = min(h, w) // 4
        cy2, cx2 = h//2, w//2
        center  = gray[cy2-margin:cy2+margin, cx2-margin:cx2+margin]
        border  = np.concatenate([
            gray[:margin,:].flatten(), gray[-margin:,:].flatten(),
            gray[:,:margin].flatten(), gray[:,-margin:].flatten()
        ])
        cm  = float(center.mean()) if center.size else 128.0
        bm  = float(border.mean()) if border.size else 128.0
        ccd = cm - bm

        hist32, _ = np.histogram(gray.flatten(), bins=32, range=(0,255))
        thr = hist32.sum() * 0.015
        peaks = 0; in_peak = False
        for v in hist32:
            if v > thr and not in_peak: peaks += 1; in_peak = True
            elif v <= thr: in_peak = False

        return {
            "chroma_diff":       float(round(chroma, 2)),
            "range":             int(mx - mn),
            "edge_density":      float(round(edge_density, 4)),
            "mean_sat":          float(round(float(sat.mean()), 4)),
            "mean_intensity":    float(round(float(gray.mean()), 1)),
            "center_brighter":   bool(ccd > 20),
            "center_corner_diff":float(round(ccd, 2)),
            "is_bimodal":        bool(peaks >= 2),
            "peak_count":        int(peaks),
            "bright_fraction":   float(round(float((gray>200).sum())/gray.size, 4)),
        }

    # ── Helpers ──────────────────────────────────────────────────────

    def _thumbnail(self, image: np.ndarray, size: int = 200) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    def _modality_label(self, m: str) -> str:
        return MODALITY_THRESHOLDS.get(m, {}).get("label", m)