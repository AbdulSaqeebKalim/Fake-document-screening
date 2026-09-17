from __future__ import annotations
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class MRZFieldOut(BaseModel):
    value: str
    check_digit: Optional[str] = None
    passed: Optional[bool] = None


class MRZOut(BaseModel):
    valid: bool
    surname: str
    given_names: str
    nationality: str
    sex: str
    issuing_state: str
    document_number: MRZFieldOut
    date_of_birth: MRZFieldOut
    date_of_expiry: MRZFieldOut
    failures: List[str] = Field(default_factory=list)


class TamperOut(BaseModel):
    score: float = Field(..., description="0-100, higher = more likely tampered")
    hotspot_bbox: Optional[Tuple[int, int, int, int]] = Field(
        None, description="x, y, w, h of the strongest anomaly region, for overlay drawing"
    )


class FaceOut(BaseModel):
    match_confidence: Optional[float] = Field(
        None, description="0-100 similarity confidence; null if no reference photo was supplied or no face detected"
    )
    document_face_bbox: Optional[Tuple[int, int, int, int]] = None
    reference_face_bbox: Optional[Tuple[int, int, int, int]] = None


class OCROut(BaseModel):
    mean_confidence: float
    low_confidence: bool
    raw_text_sample: str


class ScreeningResponse(BaseModel):
    verdict: str
    risk: int = Field(..., ge=0, le=100)
    band: str = Field(..., description="'low' | 'med' | 'high'")
    findings: List[str]
    mrz: MRZOut
    tamper: TamperOut
    face: FaceOut
    ocr: OCROut
    elapsed_seconds: float


class DemoCase(BaseModel):
    key: str
    label: str
    fields: dict
    mrz_ok: bool
    tamper: Optional[str]
    face_match: float
    risk: int
    verdict: str
    findings: List[str]
