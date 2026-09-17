"""
Serves the exact four demo scenarios the frontend prototype currently
hardcodes in its <script> block (CASES = {...}). Moving them here gives
you a single source of truth: the frontend can fetch this instead of
carrying a duplicate copy of the data, and it's a safe offline fallback
if the live model pipeline (routers/screening.py) is unavailable during
the actual judging demo (no dependency on camera/scanner hardware).
"""

from fastapi import APIRouter, HTTPException
from app.schemas import DemoCase

router = APIRouter(prefix="/api/demo-cases", tags=["demo"])

_CASES: dict[str, DemoCase] = {
    "clean": DemoCase(
        key="clean",
        label="Clean e-passport",
        fields={
            "surname": "DE ALVARES", "given": "MARISA JOY", "no": "NV3381027",
            "nat": "NARVANIAN", "dob": "14 MAR 1994", "exp": "02 JUL 2029",
        },
        mrz_ok=True, tamper=None, face_match=98.7, risk=4,
        verdict="Cleared — low risk",
        findings=[
            "MRZ checksum verified against all data fields",
            "No tampering artifacts detected in photo or text regions",
            "Facial embedding match at 98.7% confidence",
            "Document layout matches ICAO template for issuing state",
        ],
    ),
    "splice": DemoCase(
        key="splice",
        label="Photo splice attempt",
        fields={
            "surname": "KOVAC", "given": "ANDRIJA", "no": "NV5502981",
            "nat": "NARVANIAN", "dob": "22 NOV 1988", "exp": "11 JAN 2028",
        },
        mrz_ok=True, tamper="photo", face_match=41.2, risk=92,
        verdict="Flagged — likely photo substitution",
        findings=[
            "Photo region shows inconsistent JPEG re-compression (ELA anomaly)",
            "Ghosting detected around photo boundary, consistent with splicing",
            "Facial embedding distance exceeds match threshold (41.2%)",
            "Recommend escalation to secondary inspection",
        ],
    ),
    "forged_mrz": DemoCase(
        key="forged_mrz",
        label="Forged expiry date",
        fields={
            "surname": "OYELARAN", "given": "TOMISIN", "no": "NV2217450",
            "nat": "NARVANIAN", "dob": "05 AUG 1996", "exp": "02 JUL 2031",
        },
        mrz_ok=False, tamper="exp", face_match=95.4, risk=78,
        verdict="Flagged — MRZ checksum failure",
        findings=[
            "Date-of-expiry check digit does not match recomputed MRZ checksum",
            "Font kerning anomaly detected in expiry field at pixel level",
            "Facial embedding match within normal range (95.4%)",
            "Cross-check against issuing authority database recommended",
        ],
    ),
    "borderline": DemoCase(
        key="borderline",
        label="Borderline face match",
        fields={
            "surname": "HALVORSEN", "given": "INGRID", "no": "NV9013822",
            "nat": "NARVANIAN", "dob": "30 JAN 1979", "exp": "19 SEP 2027",
        },
        mrz_ok=True, tamper=None, face_match=76.3, risk=54,
        verdict="Manual review — inconclusive match",
        findings=[
            "Document and MRZ forensics clean",
            "Facial similarity below auto-clear threshold (76.3%)",
            "Possible causes: aging, lighting angle, or image quality",
            "Route to officer for manual side-by-side comparison",
        ],
    ),
}


@router.get("", response_model=list[DemoCase])
def list_demo_cases():
    return list(_CASES.values())


@router.get("/{key}", response_model=DemoCase)
def get_demo_case(key: str):
    case = _CASES.get(key)
    if case is None:
        raise HTTPException(status_code=404, detail=f"No demo case named '{key}'")
    return case
