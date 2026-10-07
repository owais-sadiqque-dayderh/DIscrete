"""Apple Vision OCR fallback when AX text is sparse (planS1 §6.1)."""

from __future__ import annotations

import io
import logging

log = logging.getLogger("discrete.capture.ocr")

# VNRequestTextRecognitionLevel: Accurate = 0, Fast = 1 (insights.md §7.12).
_ACCURATE = 0


class ExtractionFailed(RuntimeError):
    """OCR did not complete; this is not an empty successful read."""



def ocr_image(img) -> str:
    """OCR a PIL image; returns lines in Vision's reading order."""
    if img is None:
        return ""
    buf = io.BytesIO()
    # Lossless input matters for small glyphs in videos and images.
    img.convert("RGB").save(buf, "PNG")
    return ocr_jpeg(buf.getvalue())


def ocr_jpeg(jpeg: bytes) -> str:
    """Recognize encoded image bytes (JPEG or PNG)."""
    if not jpeg:
        return ""
    try:
        from Foundation import NSData
        from Vision import VNImageRequestHandler, VNRecognizeTextRequest

        handler = VNImageRequestHandler.alloc().initWithData_options_(NSData.dataWithBytes_length_(jpeg, len(jpeg)), None)
        req = VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(_ACCURATE)
        req.setUsesLanguageCorrection_(True)
        ok, err = handler.performRequests_error_([req], None)
        if not ok:
            log.debug("OCR failed: %s", err)
            raise ExtractionFailed("OCR request failed")
        lines: list[str] = []
        for obs in req.results() or []:
            cand = obs.topCandidates_(1)
            if cand:
                lines.append(str(cand[0].string()))
        return "\n".join(lines)
    except Exception as exc:
        log.debug("OCR unavailable: %s", exc)
        raise ExtractionFailed("OCR unavailable") from exc
