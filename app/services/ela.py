"""
Error Level Analysis (ELA) — real image forensics for detecting localized
digital tampering (photo splicing, re-touched text fields, copy-paste).

How it works (this is a genuine, well-established forensic technique, not
a placeholder): a photo that was saved once as a JPEG compresses uniformly
— every region loses a similar, small amount of detail. If someone edits
one region and re-saves the file, that region gets re-compressed at a
*different* generation than the rest of the image, so it responds
differently when you deliberately re-compress the whole image again and
diff it against the original. Untouched regions produce a faint, even
residual; edited regions "light up" because their error response no
longer matches their surroundings.

Pipeline:
  1. Re-save the input image as JPEG at a fixed quality.
  2. Compute the per-pixel absolute difference vs. the original.
  3. Amplify and convert to a grayscale anomaly map.
  4. Score = combination of hotspot intensity + how localized it is
     (a tampered region is a concentrated blob; uniform sensor noise or
     recompression artifacts are spread evenly across the image).
  5. Return the bounding box of the strongest anomaly region for overlay
     drawing on the frontend.

-----------------------------------------------------------------------
KNOWN LIMITATION — READ BEFORE DEMOING (this is measured, not theoretical)
-----------------------------------------------------------------------
ELA has a well-documented false positive on SHARP HIGH-CONTRAST EDGES.
JPEG always struggles to encode hard edges, so printed text, the MRZ band,
and document borders produce large residuals whether or not anything was
tampered with.

Measured on this codebase: on flat, synthetically-rendered document images
(scripts/make_test_images.py), a *clean* document scored 100/100 with the
hotspot landing on the MRZ text band — a false positive. The current
default parameters are tuned for and validated on PHOTOGRAPHIC content
(scripts/smoke_test.py: clean 18.3 vs. spliced 64.4, hotspot correctly
localized to the pasted region).

What this means in practice:
  * On real photographs/scans of real documents, the defaults are a
    reasonable starting point but are NOT calibrated for your specific
    camera/scanner.
  * You MUST calibrate against real document images before trusting the
    tamper score in a live demo. Use scripts/calibrate_ela.py, which
    sweeps `quality`, `scale` and the threshold and reports the
    clean-vs-tampered separation for YOUR images.
  * Attempts to auto-correct this with Sobel edge-energy suppression and
    with block-wise robust z-scores were both tried and both degraded
    genuine splice detection on validated photographic data, so they were
    deliberately NOT shipped. Don't re-add them without measuring.

Until calibrated, treat the tamper score as ADVISORY: the MRZ checksum
(services/mrz.py) is the deterministic, spec-exact signal and is the one
to lean on in front of judges.
"""

from __future__ import annotations
import io
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageChops
import cv2


@dataclass
class ELAResult:
    tamper_score: float          # 0-100, higher = more likely tampered
    hotspot_bbox: Optional[Tuple[int, int, int, int]]  # x, y, w, h in original image px
    hotspot_mean_intensity: float
    global_mean_intensity: float
    ela_image_png: bytes         # visualisable heatmap, ready to send/save


def _resave_jpeg(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def analyze(image: Image.Image, quality: int = 90, scale: int = 18) -> ELAResult:
    """
    Run ELA on a PIL image and return a tamper score plus the strongest
    anomaly region (for drawing an overlay box on the document viewer).
    """
    original = image.convert("RGB")
    resaved = _resave_jpeg(original, quality)

    diff = ImageChops.difference(original, resaved)
    diff_arr = np.asarray(diff).astype(np.float32)

    # Amplify so subtle differences become visible / measurable.
    amplified = np.clip(diff_arr * scale, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(amplified, cv2.COLOR_RGB2GRAY)

    global_mean = float(gray.mean())

    # Smooth then threshold to find the single most anomalous contiguous
    # region — this is what a splice or a re-touched text field looks like:
    # a blob of high residual sitting in a sea of low residual.
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh_val = max(30, int(global_mean * 2.2))
    _, mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hotspot_bbox = None
    hotspot_mean = global_mean
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= 25:  # ignore single-pixel noise specks
            x, y, w, h = cv2.boundingRect(largest)
            hotspot_bbox = (int(x), int(y), int(w), int(h))
            region = gray[y:y + h, x:x + w]
            hotspot_mean = float(region.mean()) if region.size else global_mean

    # Score blends: how much hotter the hotspot is than the image baseline,
    # and the absolute intensity of the residual. Purely uniform
    # recompression noise scores low; a genuinely edited region scores high.
    contrast_ratio = (hotspot_mean - global_mean) / (global_mean + 1e-6)
    score = 0.0
    if hotspot_bbox is not None:
        score = min(100.0, max(0.0, contrast_ratio * 28 + hotspot_mean * 0.35))
    else:
        score = min(100.0, global_mean * 0.4)

    heat = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat_img = Image.fromarray(heat_rgb)
    out_buf = io.BytesIO()
    heat_img.save(out_buf, "PNG")

    return ELAResult(
        tamper_score=round(score, 1),
        hotspot_bbox=hotspot_bbox,
        hotspot_mean_intensity=round(hotspot_mean, 2),
        global_mean_intensity=round(global_mean, 2),
        ela_image_png=out_buf.getvalue(),
    )
