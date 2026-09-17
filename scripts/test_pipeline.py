"""
Runs the exact same pipeline as POST /api/screen, but offline — no server,
no HTTP. Useful for checking that OCR/MRZ/ELA/face all behave on your test
images before you wire up the frontend, and for debugging when the live
panel returns something unexpected.

Usage:
    python scripts/test_pipeline.py                    # runs all test_images
    python scripts/test_pipeline.py path/to/doc.jpg    # a specific document
    python scripts/test_pipeline.py doc.jpg selfie.jpg # with a reference photo
"""
import os
import sys
import glob

import cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services import ela, face_match, ocr, risk_engine  # noqa: E402
from app.services import mrz as mrz_service  # noqa: E402


def run(doc_path, selfie_path=None):
    print("=" * 66)
    print("DOCUMENT:", os.path.basename(doc_path))
    if selfie_path:
        print("SELFIE:  ", os.path.basename(selfie_path))
    print("=" * 66)

    doc_bgr = cv2.imread(doc_path)
    if doc_bgr is None:
        print("  !! could not read image")
        return

    # Stage 1 - OCR
    l1, l2 = ocr.extract_mrz_lines(doc_bgr)
    ocr_full = ocr.extract_full_text(doc_bgr)
    print(f"[1] OCR      mean confidence {ocr_full.mean_confidence}")
    print(f"    MRZ L2 read: {l2!r}")

    # Stage 2 - MRZ
    parsed = mrz_service.parse_td3(l1, l2)
    print(f"[2] MRZ      valid={parsed.valid}  name={parsed.surname} {parsed.given_names}")
    for f in parsed.failures:
        print(f"    - {f}")

    # Stage 3 - ELA
    pil = Image.fromarray(cv2.cvtColor(doc_bgr, cv2.COLOR_BGR2RGB))
    e = ela.analyze(pil)
    print(f"[3] ELA      score={e.tamper_score}  hotspot={e.hotspot_bbox}")

    # Stage 4 - face
    face_pct = None
    face_reason = None
    doc_face = face_match.detect_face(doc_bgr)
    print(f"[4] FACE     document face detected={doc_face.found} bbox={doc_face.bbox}")
    if selfie_path:
        selfie_bgr = cv2.imread(selfie_path)
        if selfie_bgr is not None:
            res = face_match.compare_faces(doc_bgr, selfie_bgr)
            face_pct = res.match_confidence
            if face_pct is None:
                face_reason = "Face verification skipped — face not located"
                print("             similarity=N/A (face not detected in one/both images)")
            else:
                print(f"             similarity={face_pct}%")

    a = risk_engine.assess(
        mrz_ok=parsed.valid,
        mrz_failures=parsed.failures,
        tamper_score=e.tamper_score,
        tamper_bbox_present=e.hotspot_bbox is not None,
        face_match_pct=face_pct,
        ocr_low_confidence=ocr_full.low_confidence,
        face_not_evaluated_reason=face_reason,
    )
    print(f"\n  >> RISK {a.risk}/100  [{a.band}]  {a.verdict}")
    for f in a.findings:
        print(f"     - {f}")
    print()


def main():
    args = sys.argv[1:]
    if len(args) >= 2:
        run(args[0], args[1])
    elif len(args) == 1:
        run(args[0])
    else:
        here = os.path.join(os.path.dirname(__file__), "..", "test_images")
        docs = sorted(glob.glob(os.path.join(here, "passport_*.jpg")))
        if not docs:
            print("No test images found. Run: python scripts/make_test_images.py")
            return
        match = os.path.join(here, "selfie_match.jpg")
        diff = os.path.join(here, "selfie_different.jpg")
        for d in docs:
            sel = match if "clean" in d else (diff if "spliced" in d else None)
            run(d, sel if sel and os.path.exists(sel) else None)


if __name__ == "__main__":
    main()
