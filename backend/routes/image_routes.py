"""
backend/routes/image_routes.py   — /api/image/*
backend/routes/validation_routes.py — /api/validate/*
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
        # Handle DICOM separately
        ext = f.filename.rsplit(".", 1)[1].lower()
        if ext == "dcm":
            return _handle_dicom(f)

        # Regular image
        import io
        from PIL import Image
        pil = Image.open(io.BytesIO(f.read())).convert("RGB")
        img = np.array(pil)
        img = processor.resize_for_processing(img, max_dim=512)

        b64  = processor.numpy_to_base64(img)
        h, w = img.shape[:2]

        return jsonify({
            "status"   : "uploaded",
            "image_b64": b64,
            "width"    : w,
            "height"   : h,
            "filename" : secure_filename(f.filename),
            "modality_hint": _guess_modality(img),
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

        # Normalise to 0–255
        px  = ((px - px.min()) / (px.max() - px.min() + 1e-8) * 255).astype(np.uint8)
        if px.ndim == 2:
            import cv2
            px = cv2.cvtColor(px, cv2.COLOR_GRAY2RGB)

        px   = processor.resize_for_processing(px)
        b64  = processor.numpy_to_base64(px)

        # Extract DICOM modality tag
        modality_tag = getattr(ds, "Modality", "").upper()
        modality_map = {"CR": "xray", "DX": "xray", "MR": "mri",
                        "CT": "ct",   "XA": "bone"}
        modality     = modality_map.get(modality_tag, "xray")

        return jsonify({
            "status"       : "uploaded",
            "image_b64"    : b64,
            "width"        : px.shape[1],
            "height"       : px.shape[0],
            "filename"     : secure_filename(f.filename),
            "modality_hint": modality,
            "dicom_info"   : {
                "patient_name" : str(getattr(ds, "PatientName", "ANONYMIZED")),
                "modality"     : modality_tag,
                "study_date"   : str(getattr(ds, "StudyDate", "Unknown")),
            }
        })
    except ImportError:
        return jsonify({"error": "pydicom not installed. Run: pip install pydicom"}), 500


def _guess_modality(img: np.ndarray) -> str:
    """Quick heuristic guess at modality from pixel stats (not guaranteed)."""
    report = validator.validate(img, "xray")
    # If it passes xray, suggest xray; otherwise try others
    for m in ["xray", "mri", "bone", "ct"]:
        r = validator.validate(img, m)
        if r.status in ("pass", "warn"):
            return m
    return "xray"


# ── /api/validate/check ─────────────────────────────────────────────────────

@validation_bp.route("/check", methods=["POST"])
def check():
    """
    Validate an image against a declared modality.
    Can be called before running the full pipeline.

    Request JSON: { "image_b64": "...", "modality": "xray" }
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
        img    = processor.resize_for_processing(img, max_dim=200)
        report = validator.validate(img, modality)

        return jsonify({
            "status"   : report.status,
            "score"    : report.score,
            "message"  : report.message,
            "hint"     : report.hint,
            "checks"   : [
                {
                    "label"    : c.label,
                    "passed"   : c.passed,
                    "value"    : c.value,
                    "expected" : c.expected,
                }
                for c in report.checks
            ],
            "pixel_stats": report.pixel_stats,
        })
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500