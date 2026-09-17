"""
Face detection + similarity scoring.

HONEST SCOPE NOTE (read this before demoing to judges):
This module does two things for real:
  1. Face *detection* — locates a face in an image using OpenCV's
     Viola-Jones Haar cascade, a real, classical computer-vision detector
     shipped with opencv-python. This genuinely finds faces; it doesn't
     fake it.
  2. Face *similarity* — compares two detected faces using normalized
     grayscale histogram correlation + ORB keypoint matching. This is a
     legitimate, working similarity measure, but it is NOT a deep face
     -recognition embedding model. It will not match the accuracy of
     ArcFace/MagFace (the models named in your pitch deck) on faces with
     large pose/lighting/aging differences.

Why ship this instead of ArcFace/MagFace directly: those require dlib or
ONNX embedding models (large binary downloads, GPU-friendly builds) that
don't fit a lightweight hackathon backend. This module is deliberately
structured so a real embedding model is a drop-in replacement — implement
`FaceMatcher` with the same `.compare(img_a, img_b) -> float` interface
(see `EmbeddingFaceMatcher` stub at the bottom) and swap it in `risk_engine`
or your router without touching any other code.

For the demo: this is honestly good enough to show "clearly the same
person" vs "clearly a different person" vs "borderline / inconclusive",
which is exactly the three-way distinction your UI needs.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


@dataclass
class FaceDetection:
    found: bool
    bbox: Optional[Tuple[int, int, int, int]]  # x, y, w, h
    face_crop: Optional[np.ndarray]


def detect_face(bgr_image: np.ndarray) -> FaceDetection:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) == 0:
        return FaceDetection(found=False, bbox=None, face_crop=None)

    # Largest detected face wins (closest / most prominent).
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    crop = bgr_image[y:y + h, x:x + w]
    return FaceDetection(found=True, bbox=(int(x), int(y), int(w), int(h)), face_crop=crop)


def _normalize_face(face_bgr: np.ndarray, size: int = 128) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size))
    gray = cv2.equalizeHist(gray)
    return gray


def _histogram_similarity(a: np.ndarray, b: np.ndarray) -> float:
    hist_a = cv2.calcHist([a], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return max(0.0, corr)  # correlation in [-1, 1], clamp negative to 0


def _orb_similarity(a: np.ndarray, b: np.ndarray) -> float:
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(a, None)
    kp2, des2 = orb.detectAndCompute(b, None)
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return 0.0
    good = [m for m in matches if m.distance < 60]
    denom = min(len(kp1), len(kp2))
    return len(good) / denom if denom else 0.0


@dataclass
class FaceMatchResult:
    # None means "could not evaluate" (no face found in one or both images).
    # This is deliberately distinct from 0.0, which would mean "two faces
    # were compared and they look nothing alike". Conflating the two causes
    # a clean document with an undetectable face to be scored as an
    # impostor, which is a serious false-positive.
    match_confidence: Optional[float]  # 0-100, or None
    face_a: FaceDetection
    face_b: FaceDetection


def compare_faces(image_a_bgr: np.ndarray, image_b_bgr: np.ndarray) -> FaceMatchResult:
    """
    Detects a face in each image and returns a 0-100 similarity confidence,
    or None if a face could not be located in one or both images.
    """
    face_a = detect_face(image_a_bgr)
    face_b = detect_face(image_b_bgr)

    if not (face_a.found and face_b.found):
        return FaceMatchResult(match_confidence=None, face_a=face_a, face_b=face_b)

    norm_a = _normalize_face(face_a.face_crop)
    norm_b = _normalize_face(face_b.face_crop)

    hist_score = _histogram_similarity(norm_a, norm_b)      # 0-1
    orb_score = _orb_similarity(norm_a, norm_b)              # 0-1, uncapped-ish

    orb_score = min(1.0, orb_score * 1.6)  # ORB match ratios run low; rescale
    blended = 0.55 * hist_score + 0.45 * orb_score
    confidence = round(min(100.0, max(0.0, blended * 100)), 1)

    return FaceMatchResult(match_confidence=confidence, face_a=face_a, face_b=face_b)


class FaceMatcher:
    """Interface a production embedding model should implement to swap in."""

    def compare(self, image_a_bgr: np.ndarray, image_b_bgr: np.ndarray) -> float:
        raise NotImplementedError


class EmbeddingFaceMatcher(FaceMatcher):
    """
    Drop-in replacement stub. Wire up insightface / face_recognition /
    an ONNX ArcFace model here, keep the same `.compare()` signature, and
    point `routers/screening.py` at this class instead of `compare_faces`.
    """

    def compare(self, image_a_bgr: np.ndarray, image_b_bgr: np.ndarray) -> float:
        raise NotImplementedError(
            "Plug a real embedding model (e.g. insightface ArcFace/MagFace) in here."
        )
