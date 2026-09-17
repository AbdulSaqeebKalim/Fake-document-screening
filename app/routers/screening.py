"""
POST /api/screen — the real four-stage pipeline shown in the frontend's
stepper UI (OCR extraction -> MRZ validation -> tamper forensics -> face
verification), fused into one risk score and verdict.

Accepts:
  - document: image file of the passport/ID page (required)
  - selfie:   image file of a live capture / reference photo (optional —
              if omitted, face verification is skipped and that signal
              is simply left out of the risk calculation)
  - mrz_line1 / mrz_line2: optional raw MRZ text, if you already have it
              from a dedicated MRZ scanner and want to skip OCR entirely
"""

from __future__ import annotations
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas import (
    FaceOut, MRZFieldOut, MRZOut, OCROut, ScreeningResponse, TamperOut,
)
from app.services import ela, face_match, mrz as mrz_service, ocr, risk_engine

router = APIRouter(prefix="/api", tags=["screening"])

_MAX_UPLOAD_BYTES = 12 * 1024 * 1024  # 12 MB per image, generous for a phone/scanner capture


async def _read_image(upload: UploadFile, field_name: str) -> np.ndarray:
    raw = await upload.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{field_name} exceeds {_MAX_UPLOAD_BYTES // (1024*1024)}MB limit")
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"Could not decode {field_name} — is it a valid image file?")
    return img


@router.post("/screen", response_model=ScreeningResponse)
async def screen_document(
    document: UploadFile = File(..., description="Passport / ID page image"),
    selfie: Optional[UploadFile] = File(None, description="Live capture or reference photo, optional"),
    mrz_line1: Optional[str] = Form(None),
    mrz_line2: Optional[str] = Form(None),
):
    start = time.perf_counter()

    doc_bgr = await _read_image(document, "document")
    selfie_bgr = await _read_image(selfie, "selfie") if selfie is not None else None

    # ---- Stage 1: OCR extraction ----
    # Each stage is guarded: a failure in one signal should degrade that
    # signal only, not 500 the whole request. Real photos break OCR and
    # face detection in ways synthetic tests never hit.
    stage_errors: list[str] = []

    if mrz_line1 and mrz_line2:
        raw_l1, raw_l2 = mrz_line1, mrz_line2
    else:
        try:
            raw_l1, raw_l2 = ocr.extract_mrz_lines(doc_bgr)
        except Exception as e:
            raw_l1, raw_l2 = "", ""
            stage_errors.append(f"OCR failed: {type(e).__name__}")

    try:
        ocr_full = ocr.extract_full_text(doc_bgr)
    except Exception as e:
        ocr_full = ocr.OCRResult(text="", mean_confidence=0.0, low_confidence=True)
        stage_errors.append(f"Full-text OCR failed: {type(e).__name__}")

    # ---- Stage 2: MRZ checksum validation ----
    parsed = mrz_service.parse_td3(raw_l1, raw_l2)

    # ---- Stage 3: Tamper forensics (ELA) ----
    from PIL import Image
    try:
        doc_rgb_pil = Image.fromarray(cv2.cvtColor(doc_bgr, cv2.COLOR_BGR2RGB))
        ela_result = ela.analyze(doc_rgb_pil)
        tamper_score = ela_result.tamper_score
        tamper_bbox = ela_result.hotspot_bbox
    except Exception as e:
        tamper_score, tamper_bbox = 0.0, None
        stage_errors.append(f"Tamper analysis failed: {type(e).__name__}")

    # ---- Stage 4: Face verification ----
    face_out = FaceOut(match_confidence=None, document_face_bbox=None, reference_face_bbox=None)
    face_match_pct: Optional[float] = None
    face_skip_reason: Optional[str] = None
    try:
        doc_face = face_match.detect_face(doc_bgr)
        face_out.document_face_bbox = doc_face.bbox
        if selfie_bgr is not None:
            result = face_match.compare_faces(doc_bgr, selfie_bgr)
            face_match_pct = result.match_confidence
            face_out.match_confidence = face_match_pct
            face_out.reference_face_bbox = result.face_b.bbox
            if face_match_pct is None:
                # Could not locate a face in one or both images. This is NOT a
                # mismatch - report it as an unevaluated signal rather than
                # letting it inflate the risk score.
                if not result.face_a.found and not result.face_b.found:
                    face_skip_reason = "Face verification skipped — no face located in either image"
                elif not result.face_a.found:
                    face_skip_reason = "Face verification skipped — no face located on the document image"
                else:
                    face_skip_reason = "Face verification skipped — no face located in the reference photo"
    except Exception as e:
        face_skip_reason = f"Face verification skipped — analysis failed ({type(e).__name__})"
        stage_errors.append(f"Face analysis failed: {type(e).__name__}")

    # ---- Fuse into a risk verdict ----
    assessment = risk_engine.assess(
        mrz_ok=parsed.valid,
        mrz_failures=parsed.failures,
        tamper_score=tamper_score,
        tamper_bbox_present=tamper_bbox is not None,
        face_match_pct=face_match_pct,
        ocr_low_confidence=ocr_full.low_confidence,
        face_not_evaluated_reason=face_skip_reason,
    )

    for se in stage_errors:
        assessment.findings.append("Pipeline warning — " + se)

    elapsed = round(time.perf_counter() - start, 3)

    return ScreeningResponse(
        verdict=assessment.verdict,
        risk=assessment.risk,
        band=assessment.band,
        findings=assessment.findings,
        mrz=MRZOut(
            valid=parsed.valid,
            surname=parsed.surname,
            given_names=parsed.given_names,
            nationality=parsed.nationality,
            sex=parsed.sex,
            issuing_state=parsed.issuing_state,
            document_number=MRZFieldOut(
                value=parsed.document_number.value,
                check_digit=parsed.document_number.check_digit,
                passed=parsed.document_number.passed,
            ),
            date_of_birth=MRZFieldOut(
                value=parsed.date_of_birth.value,
                check_digit=parsed.date_of_birth.check_digit,
                passed=parsed.date_of_birth.passed,
            ),
            date_of_expiry=MRZFieldOut(
                value=parsed.date_of_expiry.value,
                check_digit=parsed.date_of_expiry.check_digit,
                passed=parsed.date_of_expiry.passed,
            ),
            failures=parsed.failures,
        ),
        tamper=TamperOut(score=tamper_score, hotspot_bbox=tamper_bbox),
        face=face_out,
        ocr=OCROut(
            mean_confidence=ocr_full.mean_confidence,
            low_confidence=ocr_full.low_confidence,
            raw_text_sample=ocr_full.text[:200],
        ),
        elapsed_seconds=elapsed,
    )
