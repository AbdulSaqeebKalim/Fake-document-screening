"""
OCR extraction using Tesseract (via pytesseract) — real text extraction,
not mocked. Two things happen here:

1. `extract_mrz_lines()` crops the bottom band of a passport-style image
   (where the Machine Readable Zone lives on every ICAO 9303 document),
   binarizes it, and runs Tesseract restricted to the MRZ character set
   (A-Z, 0-9, '<'). This is the standard approach used by real MRZ readers.

2. `extract_full_text()` runs general OCR over the whole document image,
   for surfacing raw extracted text / confidence when there's no clean
   MRZ band (visas, ID cards, non-standard layouts).

Note: on a real scanner feed, MRZ-band localization is usually done with
a small object-detection model; here we use the reliable heuristic that
the MRZ occupies the bottom ~22-28% of a properly cropped passport image,
which is sufficient for a hackathon demo pipeline and easy to replace with
a learned detector later.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import pytesseract

_MRZ_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"


@dataclass
class OCRResult:
    text: str
    mean_confidence: float
    low_confidence: bool


def _preprocess_for_ocr(bgr_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 60, 60)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_mrz_lines(bgr_image: np.ndarray) -> Tuple[str, str]:
    """
    Crop the bottom MRZ band of a passport-style image and OCR it into
    two 44-character candidate lines. Returns raw (possibly imperfect)
    strings — callers should feed them to services.mrz.parse_td3, which
    is tolerant of minor padding differences.
    """
    h, w = bgr_image.shape[:2]
    band = bgr_image[int(h * 0.72):h, 0:w]
    processed = _preprocess_for_ocr(band)

    config = (
        f"--psm 6 -c tessedit_char_whitelist={_MRZ_WHITELIST}"
    )
    raw = pytesseract.image_to_string(processed, config=config)
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    if len(lines) >= 2:
        return lines[-2], lines[-1]
    if len(lines) == 1:
        # Tesseract sometimes merges both MRZ lines into one blob.
        joined = lines[0]
        mid = len(joined) // 2
        return joined[:mid], joined[mid:]
    return "", ""


def extract_full_text(bgr_image: np.ndarray) -> OCRResult:
    processed = _preprocess_for_ocr(bgr_image)
    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)

    words: List[str] = []
    confidences: List[float] = []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        text = text.strip()
        if not text:
            continue
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if c < 0:
            continue
        words.append(text)
        confidences.append(c)

    full_text = " ".join(words)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return OCRResult(text=full_text, mean_confidence=round(mean_conf, 1), low_confidence=mean_conf < 55)
