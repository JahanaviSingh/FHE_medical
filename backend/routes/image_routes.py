"""
backend/routes/image_routes.py   — /api/image/*
backend/routes/validation_routes.py — /api/validate/*

Fixed issues:
  1. /api/validate/check: resize max_dim raised 200→512 so physics checks
     (isotropy, extreme_frac, bone signal) see the same resolution as the
     image actually uploaded — 200px made the dark border strip dominant.
  2. _guess_modality(): removed — it called validator.validate() 4× on every
     upload (slow, noisy logs). Replaced with a fast pixel-stats heuristic.
"""

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os, uuid, logging
import numpy as np

from backend.utils.image_processor import ImageProcessor
from backend.utils.validator        import MedicalImageValidator

image_bp      = Blueprint("image",      __name__)
validation_bp = Blueprint("validation", __name__)
processor     = ImageProcessor()
validator     = MedicalImageValidator()
logger        = logging.getLogger(__name__)

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff", "dcm"}

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ── /api/image/upload ───────────────────────────────────────────────────────

@image_bp.route("/upload", methods=["POST"])
def upload():
    """
    Accept multipart/form-data file upload.
    Returns base64 preview + basic image info.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not _allowed(f.filename):
        return jsonify({"error": f"File type not allowed. Use: {ALLOWED_EXT}"}), 415

    try:
        ext = f.filename.rsplit(".", 1)[1].lower()
        if ext == "dcm":
            return _handle_dicom(f)

        import io
        from PIL import Image
        pil = Image.open(io.BytesIO(f.read())).convert("RGB")
        img = np.array(pil)
        img = processor.resize_for_processing(img, max_dim=512)

        b64  = processor.numpy_to_base64(img)
        h, w = img.shape[:2]

        return jsonify({
            "status"        : "uploaded",
            "image_b64"     : b64,
            "width"         : w,
            "height"        : h,
            "filename"      : secure_filename(f.filename),
            "modality_hint" : _guess_modality_fast(img),
        })

    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500


def _handle_dicom(f) -> tuple:
    """Read DICOM file using pydicom and extract pixel array."""
    try:
        import pydicom, io
        ds  = pydicom.dcmread(io.BytesIO(f.read()))
        px  = ds.pixel_array.astype(np.float32)

        px  = ((px - px.min()) / (px.max() - px.min() + 1e-8) * 255).astype(np.uint8)
        if px.ndim == 2:
            import cv2
            px = cv2.cvtColor(px, cv2.COLOR_GRAY2RGB)

        px   = processor.resize_for_processing(px)
        b64  = processor.numpy_to_base64(px)

        modality_tag = getattr(ds, "Modality", "").upper()
        modality_map = {"CR": "xray", "DX": "xray", "MR": "mri",
                        "CT": "ct",   "XA": "bone"}
        modality     = modality_map.get(modality_tag, "xray")

        return jsonify({
            "status"        : "uploaded",
            "image_b64"     : b64,
            "width"         : px.shape[1],
            "height"        : px.shape[0],
            "filename"      : secure_filename(f.filename),
            "modality_hint" : modality,
            "dicom_info"    : {
                "patient_name" : str(getattr(ds, "PatientName", "ANONYMIZED")),
                "modality"     : modality_tag,
                "study_date"   : str(getattr(ds, "StudyDate", "Unknown")),
            }
        })
    except ImportError:
        return jsonify({"error": "pydicom not installed. Run: pip install pydicom"}), 500


def _guess_modality_fast(img: np.ndarray) -> str:
    """
    Fast pixel-stats heuristic — no validator calls, no ML, no extra latency.

    Rules (in priority order):
      1. Colour image (chroma > 30)           → non-medical, default "xray"
      2. Large dark background + bright spots → bone or xray
         - centre NOT brighter than edges     → bone  (limb, off-centre anatomy)
         - centre brighter than edges         → xray  (chest, centred field)
      3. Blue tint (mean_b > mean_r + 8)      → mri or ct
         - high edge density (>0.08)          → ct
         - else                               → mri
      4. Otherwise                            → xray
    """
    import cv2
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    small = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    b = small[:, :, 0].astype(float)
    g = small[:, :, 1].astype(float)
    r = small[:, :, 2].astype(float)
    gray = (r * 0.299 + g * 0.587 + b * 0.114).astype(np.uint8)

    chroma   = abs(r.mean() - g.mean()) + abs(g.mean() - b.mean()) + abs(r.mean() - b.mean())
    mean_b   = b.mean()
    mean_r   = r.mean()

    h, w     = gray.shape
    margin   = min(h, w) // 4
    center   = gray[h//2 - margin : h//2 + margin, w//2 - margin : w//2 + margin]
    border   = np.concatenate([
        gray[:margin, :].flatten(), gray[-margin:, :].flatten(),
        gray[:, :margin].flatten(), gray[:, -margin:].flatten()
    ])
    cb_ratio = float(center.mean() + 1) / (float(border.mean()) + 1)

    edges        = cv2.Canny(gray, 40, 120)
    edge_density = float(edges.sum()) / (255.0 * h * w)
    dark_frac    = float((gray < 20).mean())

    if chroma > 30:
        return "xray"                                    # colour → not medical

    if mean_b > mean_r + 8:                              # blue-tinted scanner
        return "ct" if edge_density > 0.08 else "mri"

    if dark_frac > 0.35 and cb_ratio < 1.3:             # off-centre anatomy
        return "bone"

    return "xray"


# ── /api/validate/check ─────────────────────────────────────────────────────

@validation_bp.route("/check", methods=["POST"])
def check():
    """
    Validate an image against a declared modality.

    Request JSON: { "image_b64": "...", "modality": "xray" }

    FIX: resize to max_dim=512 (was 200). At 200px the dark collimation
    border on bone/limb X-rays dominates the gradient histogram and kills
    the isotropy check. 512px matches the resolution used by the pipeline.
    """
    data      = request.get_json(silent=True) or {}
    image_b64 = data.get("image_b64")
    modality  = data.get("modality", "xray")

    if not image_b64:
        return jsonify({"error": "image_b64 required"}), 400

    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]

    try:
        img    = processor.base64_to_numpy(image_b64)
        img    = processor.resize_for_processing(img, max_dim=512)   # was 200
        report = validator.validate(img, modality)

        return jsonify({
            "status"      : report.status,
            "score"       : report.score,
            "message"     : report.message,
            "hint"        : report.hint,
            "checks"      : [
                {
                    "label"   : c.label,
                    "passed"  : c.passed,
                    "value"   : c.value,
                    "expected": c.expected,
                }
                for c in report.checks
            ],
            "pixel_stats" : report.pixel_stats,
        })
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500
