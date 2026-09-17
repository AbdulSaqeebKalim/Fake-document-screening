"""
ICAO Doc 9303 Machine Readable Zone (MRZ) parsing and checksum validation.

This is a *real*, spec-accurate implementation of the check-digit algorithm
used on passports, visas and ID cards worldwide (TD3 / passport format:
two 44-character lines). It is not a simulation — feed it real MRZ text
(from a scan, or typed in) and it will tell you, with certainty, whether
the check digits are internally consistent.

Reference: ICAO Doc 9303, Part 4, Section 4.9 (Check Digits).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# Weight sequence used by the ICAO 9303 check-digit algorithm, repeating.
_WEIGHTS = (7, 3, 1)

# Character -> numeric value mapping. '<' is filler (=0), digits are
# themselves, letters are 10-35 in alphabetical order.
def _char_value(ch: str) -> int:
    if ch == "<":
        return 0
    if ch.isdigit():
        return int(ch)
    if ch.isalpha():
        return ord(ch.upper()) - ord("A") + 10
    # Anything else is not a valid MRZ character. This happens routinely
    # with real OCR on a real photo (stray punctuation, noise glyphs), so
    # we must NOT raise here - a malformed read is a validation failure to
    # report, not a server crash. Treat it as filler; the check digits will
    # fail to match, which is the correct outcome for an unreadable MRZ.
    return 0


def is_valid_mrz_string(data: str) -> bool:
    """True if every character is in the MRZ alphabet (A-Z, 0-9, '<')."""
    return all(c == "<" or c.isdigit() or (c.isalpha() and c.isascii()) for c in data)


def compute_check_digit(data: str) -> int:
    """Compute the ICAO 9303 check digit for a string of MRZ data."""
    total = 0
    for i, ch in enumerate(data):
        total += _char_value(ch) * _WEIGHTS[i % 3]
    return total % 10


def verify_check_digit(data: str, check_digit: str) -> bool:
    """Return True if check_digit matches the recomputed digit for data."""
    if not check_digit.isdigit():
        # A '<' in a check-digit position is treated as a non-match, since
        # a genuine document always carries a numeric check digit here.
        return False
    return compute_check_digit(data) == int(check_digit)


@dataclass
class MRZField:
    name: str
    value: str
    check_digit: Optional[str] = None
    passed: Optional[bool] = None


@dataclass
class MRZResult:
    valid: bool
    document_number: MRZField
    date_of_birth: MRZField
    date_of_expiry: MRZField
    composite: MRZField
    surname: str = ""
    given_names: str = ""
    nationality: str = ""
    sex: str = ""
    issuing_state: str = ""
    optional_data: str = ""
    failures: list = field(default_factory=list)
    raw_line1: str = ""
    raw_line2: str = ""


def parse_td3(line1: str, line2: str) -> MRZResult:
    """
    Parse and validate a TD3 (passport) MRZ: two lines of 44 characters.

    Line 1: P<ISSUING_STATE<SURNAME<<GIVEN<NAMES<<<<<<<<<<<<<<<<<<<<<<<<
    Line 2: DOCNO<CHECK<NATIONALITY<DOB<CHECK<SEX<EXPIRY<CHECK<OPTIONAL<COMPOSITE_CHECK
    """
    line1 = line1.strip().upper().ljust(44, "<")[:44]
    line2 = line2.strip().upper().ljust(44, "<")[:44]
    failures: list[str] = []

    issuing_state = line1[2:5].replace("<", "")
    names_part = line1[5:]
    if "<<" in names_part:
        surname_raw, given_raw = names_part.split("<<", 1)
    else:
        surname_raw, given_raw = names_part, ""
    surname = surname_raw.replace("<", " ").strip()
    given_names = given_raw.replace("<", " ").strip()

    doc_no_field = line2[0:9]
    doc_no_check = line2[9]
    nationality = line2[10:13].replace("<", "")
    dob_field = line2[13:19]
    dob_check = line2[19]
    sex = line2[20]
    expiry_field = line2[21:27]
    expiry_check = line2[27]
    optional_data = line2[28:42].replace("<", "").strip()
    composite_check = line2[43] if len(line2) > 43 else ""

    doc_no_ok = verify_check_digit(doc_no_field, doc_no_check)
    dob_ok = verify_check_digit(dob_field, dob_check)
    expiry_ok = verify_check_digit(expiry_field, expiry_check)

    composite_data = (
        line2[0:10] + line2[13:20] + line2[21:43]
    )
    composite_ok = verify_check_digit(composite_data, composite_check)

    if not doc_no_ok:
        failures.append("Document number check digit mismatch")
    if not dob_ok:
        failures.append("Date-of-birth check digit mismatch")
    if not expiry_ok:
        failures.append("Date-of-expiry check digit mismatch")
    if not composite_ok:
        failures.append("Composite (final) check digit mismatch")

    charset_ok = is_valid_mrz_string(line1) and is_valid_mrz_string(line2)
    if not charset_ok:
        failures.append(
            "MRZ contains characters outside the ICAO alphabet — the zone was "
            "likely misread (poor scan quality, glare, or not a TD3 document)"
        )

    result = MRZResult(
        valid=doc_no_ok and dob_ok and expiry_ok and composite_ok and charset_ok,
        document_number=MRZField("document_number", doc_no_field.replace("<", ""), doc_no_check, doc_no_ok),
        date_of_birth=MRZField("date_of_birth", dob_field, dob_check, dob_ok),
        date_of_expiry=MRZField("date_of_expiry", expiry_field, expiry_check, expiry_ok),
        composite=MRZField("composite", composite_data, composite_check, composite_ok),
        surname=surname,
        given_names=given_names,
        nationality=nationality,
        sex=sex,
        issuing_state=issuing_state,
        optional_data=optional_data,
        failures=failures,
        raw_line1=line1,
        raw_line2=line2,
    )
    return result
