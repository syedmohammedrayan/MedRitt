"""Regression tests for scan-type upload validation."""

import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from PIL import Image

from routers.scan import _enforce_scan_type_verification, _validate_scan_matches_selected_type
from services.scan_type_verifier import ScanTypeVerification, ScanTypeVerifier


def grayscale_test_image() -> Image.Image:
    """Return a structured, non-flat grayscale image accepted by both models."""
    y, x = np.mgrid[0:256, 0:256]
    pixels = 35 + 150 * np.exp(-(((x - 128) / 82) ** 2 + ((y - 128) / 96) ** 2))
    pixels += 24 * np.sin(x / 9.0) * np.cos(y / 13.0)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="L").convert("RGB")


class ScanValidationTests(unittest.TestCase):
    def test_authoritative_dicom_modality_mismatch_is_still_rejected(self):
        with self.assertRaises(HTTPException):
            _validate_scan_matches_selected_type(grayscale_test_image(), "brain_tumor", "DX")
        with self.assertRaises(HTTPException):
            _validate_scan_matches_selected_type(grayscale_test_image(), "pneumonia", "MR")

    def test_blank_image_is_rejected_for_both_models(self):
        blank = Image.new("RGB", (256, 256), (80, 80, 80))
        for scan_type, modality in (("brain_tumor", "MRI"), ("pneumonia", "X-ray")):
            with self.subTest(scan_type=scan_type), self.assertRaises(HTTPException):
                _validate_scan_matches_selected_type(blank, scan_type, modality)

    def test_colour_photo_is_rejected_for_grayscale_models(self):
        # We expect color photos to be rejected if they are too colorful and we strictly want grayscale
        # Actually for skin_cancer we don't, but let's test brain_tumor and pneumonia (grayscale scans)
        y, x = np.mgrid[0:256, 0:256]
        colour = np.stack((x, y, (x + y) % 256), axis=-1).astype(np.uint8)
        image = Image.fromarray(colour, mode="RGB")
        for scan_type, modality in (("brain_tumor", "MRI"), ("pneumonia", "X-ray")):
            with self.subTest(scan_type=scan_type), self.assertRaises(HTTPException):
                _validate_scan_matches_selected_type(image, scan_type, modality)

if __name__ == "__main__":
    unittest.main()
