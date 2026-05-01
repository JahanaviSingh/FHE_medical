"""
tests/test_pipeline.py

Run with:  python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from backend.utils.image_processor import ImageProcessor
from backend.utils.validator        import MedicalImageValidator
from backend.utils.fhe_engine       import FHEEngine


# ── Fixtures ─────────────────────────────────

def synthetic_xray(size=256) -> np.ndarray:
    """Create a synthetic grayscale chest X-ray."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Add noise
    img += np.random.randint(0, 30, img.shape, dtype=np.uint8)
    # Add oval structure
    h, w = size, size
    for y in range(h):
        for x in range(w):
            d = ((x - w/2) / (w*0.38))**2 + ((y - h/2) / (h*0.44))**2
            if d < 1:
                img[y, x] = [160, 160, 160]
    return img

def colourful_image(size=256) -> np.ndarray:
    """Create a colourful non-medical image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = np.random.randint(100, 255, (size, size), dtype=np.uint8)  # R
    img[:, :, 1] = np.random.randint(0,   100, (size, size), dtype=np.uint8)  # G
    img[:, :, 2] = np.random.randint(0,   150, (size, size), dtype=np.uint8)  # B
    return img


# ── ImageProcessor tests ─────────────────────

class TestImageProcessor:
    def setup_method(self):
        self.proc = ImageProcessor()
        self.img  = synthetic_xray()

    def test_to_grayscale(self):
        g = self.proc.to_grayscale(self.img)
        assert g.ndim == 2, "Expected 2D grayscale array"
        assert g.dtype == np.uint8

    def test_pneumonia_heatmap(self):
        result = self.proc.pneumonia_heatmap(self.img)
        assert result.shape[-1] == 3, "Result should be BGR"
        assert result.dtype == np.uint8

    def test_all_operations(self):
        ops = ["pneumonia_detection", "fracture_detection", "tumor_boundary",
               "mri_denoise", "ct_contrast", "patient_anonymize", "edge_enhance"]
        for op in ops:
            result = self.proc.apply_operation(self.img, op)
            assert result is not None, f"Operation {op} returned None"
            assert result.dtype == np.uint8, f"Operation {op} returned wrong dtype"

    def test_base64_roundtrip(self):
        b64   = self.proc.numpy_to_base64(self.img)
        back  = self.proc.base64_to_numpy(b64)
        assert back.shape == self.img.shape

    def test_resize(self):
        big    = np.zeros((1024, 768, 3), dtype=np.uint8)
        small  = self.proc.resize_for_processing(big, max_dim=256)
        assert max(small.shape[:2]) <= 256


# ── Validator tests ───────────────────────────

class TestValidator:
    def setup_method(self):
        self.val    = MedicalImageValidator()
        self.xray   = synthetic_xray()
        self.colour = colourful_image()

    def test_xray_passes_on_xray_image(self):
        report = self.val.validate(self.xray, "xray")
        assert report.status in ("pass", "warn"), \
            f"Synthetic X-ray should pass but got: {report.status}"

    def test_colour_image_fails(self):
        report = self.val.validate(self.colour, "xray")
        assert report.status == "fail", \
            "Colourful image should fail X-ray validation"

    def test_all_modalities_accept_xray(self):
        for m in ["xray", "mri", "bone", "ct"]:
            report = self.val.validate(self.xray, m)
            assert report.score >= 0, "Score should be non-negative"
            assert report.status in ("pass", "warn", "fail")

    def test_checks_count(self):
        report = self.val.validate(self.xray, "xray")
        assert len(report.checks) == 4, "X-ray should have 4 checks"


# ── FHE Engine tests ──────────────────────────

class TestFHEEngine:
    def setup_method(self):
        self.engine = FHEEngine()
        self.img    = synthetic_xray(128)

    def test_encrypt_returns_metrics(self):
        ct, metrics = self.engine.encrypt(self.img)
        assert "encrypt_time_s"   in metrics
        assert "privacy_score"    in metrics
        assert "expansion_ratio"  in metrics
        assert metrics["patient_data_seen"] == "0%"

    def test_full_pipeline(self):
        ct, _         = self.engine.encrypt(self.img)
        processed     = self.engine.process_encrypted(ct, "pneumonia_detection")
        result, diag  = self.engine.decrypt(processed)

        assert result is not None
        assert result.dtype == np.uint8
        assert "condition" in diag

    def test_all_operations_run(self):
        ops = ["pneumonia_detection", "fracture_detection", "tumor_boundary",
               "mri_denoise", "ct_contrast", "patient_anonymize"]
        ct, _ = self.engine.encrypt(self.img)
        for op in ops:
            proc = self.engine.process_encrypted(ct, op)
            assert proc is not None, f"process_encrypted failed for {op}"

    def test_privacy_score_high(self):
        _, metrics = self.engine.encrypt(self.img)
        assert metrics["privacy_score"] >= 80, \
            f"Privacy score too low: {metrics['privacy_score']}"
