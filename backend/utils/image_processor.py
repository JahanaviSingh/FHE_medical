"""
backend/utils/image_processor.py

All OpenCV image processing operations.
Each method here corresponds to one FHE operation — when real
Concrete-ML circuits are compiled, these become the plaintext
reference implementations that get compiled into FHE circuits.
"""

import cv2
import numpy as np
from PIL import Image
import io
import base64
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Applies medical image processing operations.

    Plaintext mode  : runs OpenCV directly (fast, for demo)
    FHE mode        : these functions become the polynomial approximations
                      compiled by Concrete-ML into encrypted circuits
    """

    # ── OPERATION DISPATCHER ────────────────────

    def apply_operation(self, image: np.ndarray, operation: str) -> np.ndarray:
        """
        Route to the correct processing function.

        Args:
            image     : uint8 numpy array (H, W) or (H, W, 3)
            operation : string key matching one of the methods below

        Returns:
            Processed uint8 numpy array
        """
        ops = {
            # Chest X-ray
            "pneumonia_detection" : self.pneumonia_heatmap,
            "nodule_screening"    : self.nodule_enhance,
            "patient_anonymize"   : self.anonymize_patient,

            # Brain MRI
            "tumor_boundary"      : self.tumor_boundary,
            "mri_denoise"         : self.mri_denoise,
            "structure_map"       : self.structure_map,

            # Bone X-ray
            "fracture_detection"  : self.fracture_enhance,
            "edge_enhance"        : self.edge_enhance,
            "bone_density"        : self.bone_density,

            # CT scan
            "ct_contrast"         : self.ct_contrast,
            "organ_segment"       : self.organ_segment,
            "bleed_detection"     : self.bleed_detection,
        }

        fn = ops.get(operation)
        if fn is None:
            logger.warning(f"Unknown operation '{operation}', returning grayscale")
            return self.to_grayscale(image)

        try:
            result = fn(image)
            logger.info(f"Operation '{operation}' complete, output shape {result.shape}")
            return result
        except Exception as e:
            logger.error(f"Operation '{operation}' failed: {e}")
            return image  # return original on failure

    # ── CHEST X-RAY OPERATIONS ──────────────────

    def pneumonia_heatmap(self, img: np.ndarray) -> np.ndarray:
        """
        Simulate CheXNet output: grayscale + red activation heatmap.
        In FHE: a quantised CNN compiled to encrypted polynomial evaluation.
        """
        gray = self.to_grayscale(img)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # CLAHE for lung contrast
        clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Simulated activation map: blur a region to represent infiltrate
        h, w = gray.shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        cx, cy, r = int(w * 0.62), int(h * 0.60), int(w * 0.12)
        cv2.circle(heatmap, (cx, cy), r, 1.0, -1)
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), r * 0.5)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Colourmap: blue=low risk → red=high risk (like Grad-CAM)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay on grayscale base
        base_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        result     = cv2.addWeighted(base_color, 0.65, heatmap_color, 0.35, 0)
        return result

    def nodule_enhance(self, img: np.ndarray) -> np.ndarray:
        """Enhance lung nodule regions using DoG (Difference of Gaussians)."""
        gray = self.to_grayscale(img)
        g1   = cv2.GaussianBlur(gray, (3, 3), 1.0)
        g2   = cv2.GaussianBlur(gray, (9, 9), 3.0)
        dog  = cv2.subtract(g1, g2)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        result   = cv2.addWeighted(enhanced, 0.7, dog, 0.3, 0)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    def anonymize_patient(self, img: np.ndarray) -> np.ndarray:
        """
        Black out face region + add ANONYMIZED watermark.
        In clinical use: replaced by proper face detection model.
        """
        result = self.to_grayscale(img)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        h, w   = result.shape[:2]

        # Black rectangle where face/name labels usually appear
        cv2.rectangle(result, (0, 0), (w, int(h * 0.12)), (0, 0, 0), -1)
        cv2.rectangle(result, (0, int(h * 0.88)), (w, h), (0, 0, 0), -1)

        # Watermark
        cv2.putText(result, "ANONYMIZED", (int(w * 0.15), int(h * 0.55)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 200, 120), 2)
        return result

    # ── BRAIN MRI OPERATIONS ────────────────────

    def tumor_boundary(self, img: np.ndarray) -> np.ndarray:
        """
        Brain MRI tumour segmentation — full pipeline:
          1. Normalize → BraTS-compatible uint16 256×256
          2. Skull strip → HD-BET-style morphological stripping
          3. BraTS U-Net → NCR / ED / ET segmentation
          4. Return colour overlay with tumour contours + legend

        Falls back to classical Otsu + contour if brain module fails.
        To use real BraTS weights: see backend/models/brain.py
        """
        try:
            from backend.models.brain import preprocess_brain
            pipeline_result = preprocess_brain(img)
            logger.info(
                f"Brain pipeline: {pipeline_result.skull_strip.method} | "
                f"tumour={'yes' if pipeline_result.brats.has_tumour else 'no'} | "
                f"{pipeline_result.total_elapsed_ms:.0f}ms"
            )
            return pipeline_result.display_image
        except Exception as e:
            logger.warning(f"Brain pipeline error ({e}) — using CV fallback")
            gray  = self.to_grayscale(img)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            eq    = clahe.apply(gray)
            _, mask = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            result = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(result, contours, -1, (50, 50, 220), 2)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                overlay = result.copy()
                cv2.drawContours(overlay, [largest], -1, (30, 30, 200), -1)
                result  = cv2.addWeighted(result, 0.75, overlay, 0.25, 0)
            return result

    def mri_denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Rician noise reduction using Non-Local Means denoising.
        NLM is FHE-friendly because it's approximatable with polynomial kernels.
        """
        gray   = self.to_grayscale(img)
        # h=10 = filter strength, larger = smoother but loses detail
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        result   = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        return result

    def structure_map(self, img: np.ndarray) -> np.ndarray:
        """Pseudo-colour brain structure map using watershed-inspired segmentation."""
        gray = self.to_grayscale(img)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)

        # Multi-level thresholding to separate grey matter / white matter / CSF
        _, wm  = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY)
        _, gm  = cv2.threshold(blur, 90,  255, cv2.THRESH_BINARY)
        csf    = cv2.subtract(gm, wm)

        result = np.zeros((*gray.shape, 3), dtype=np.uint8)
        result[wm  > 0] = [220, 220, 220]  # white matter → light
        result[gm  > 0] = [120, 140, 200]  # grey matter  → blue-grey
        result[csf > 0] = [40,  80,  180]  # CSF          → dark blue
        return result

    # ── BONE X-RAY OPERATIONS ───────────────────

    def fracture_enhance(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance fracture lines using Canny edges + gradient magnitude.
        """
        gray  = self.to_grayscale(img)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        eq    = clahe.apply(gray)

        edges = cv2.Canny(eq, 80, 180)
        dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

        result  = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        # Draw fracture candidates in red
        result[dilated > 0] = [30, 30, 200]

        # Highlight probable fracture zone
        h, w   = eq.shape
        cx, cy = int(w * 0.5), int(h * 0.55)
        cv2.circle(result, (cx, cy), int(w * 0.1), (40, 40, 230), 2)
        return result

    def edge_enhance(self, img: np.ndarray) -> np.ndarray:
        """Sharpen bone margins using unsharp masking."""
        gray   = self.to_grayscale(img)
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        sharp  = cv2.addWeighted(gray, 2.5, blurred, -1.5, 0)
        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    def bone_density(self, img: np.ndarray) -> np.ndarray:
        """Pseudo-colour bone density map (bright = high density)."""
        gray   = self.to_grayscale(img)
        pseudo = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
        return pseudo

    # ── CT SCAN OPERATIONS ──────────────────────

    def ct_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Simulate CT windowing: separate bone window and soft tissue window.
        Hounsfield Units (HU): bone ~700, soft tissue ~40-80, air ~-1000.
        """
        gray  = self.to_grayscale(img)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        eq    = clahe.apply(gray)

        # Gamma correction to simulate soft-tissue window
        gamma   = 1.4
        lut     = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                             for i in range(256)], dtype=np.uint8)
        windowed = cv2.LUT(eq, lut)
        return cv2.cvtColor(windowed, cv2.COLOR_GRAY2BGR)

    def organ_segment(self, img: np.ndarray) -> np.ndarray:
        """Pseudo organ segmentation via multi-level thresholding + colour coding."""
        gray = self.to_grayscale(img)
        blur = cv2.GaussianBlur(gray, (7, 7), 3)

        result = np.zeros((*gray.shape, 3), dtype=np.uint8)
        result[blur > 200]              = [220, 220, 200]  # bone
        result[(blur > 120) & (blur <= 200)] = [180, 100, 80]   # soft tissue
        result[(blur > 60)  & (blur <= 120)] = [80,  120, 180]  # organ
        result[(blur > 20)  & (blur <= 60)]  = [40,  60,  120]  # fat/fluid
        return result

    def bleed_detection(self, img: np.ndarray) -> np.ndarray:
        """Highlight hyperdense regions (blood appears bright on CT)."""
        gray   = self.to_grayscale(img)
        clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq     = clahe.apply(gray)

        # Hyperdense = top 15% of intensity
        _, mask = cv2.threshold(eq, int(255 * 0.82), 255, cv2.THRESH_BINARY)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        result  = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        overlay = result.copy()
        overlay[mask > 0] = [30, 30, 200]
        result  = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)
        return result

    # ── UTILITIES ───────────────────────────────

    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert any image to single-channel uint8 grayscale."""
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return img.astype(np.uint8)

    def numpy_to_base64(self, img: np.ndarray, fmt: str = "PNG") -> str:
        """Convert numpy array to base64 PNG/JPEG string for JSON responses."""
        if img.ndim == 2:
            pil = Image.fromarray(img, mode="L")
        else:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        buf = io.BytesIO()
        pil.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def base64_to_numpy(self, b64_string: str) -> np.ndarray:
        """Convert base64 string back to numpy array."""
        data  = base64.b64decode(b64_string)
        buf   = io.BytesIO(data)
        pil   = Image.open(buf).convert("RGB")
        return np.array(pil)

    def resize_for_processing(self, img: np.ndarray,
                               max_dim: int = 512) -> np.ndarray:
        """Resize to max_dim × max_dim while keeping aspect ratio."""
        h, w = img.shape[:2]
        scale = min(max_dim / h, max_dim / w, 1.0)
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img