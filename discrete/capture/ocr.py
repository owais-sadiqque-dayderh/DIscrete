"""Apple Vision OCR fallback when AX text is sparse."""

from __future__ import annotations

import logging

log = logging.getLogger("discrete.capture.ocr")


def ocr_jpeg(jpeg: bytes) -> str:
    if not jpeg:
        return ""
    try:
        from Vision import VNRecognizeTextRequest, VNImageRequestHandler
        from Foundation import NSData

        handler = VNImageRequestHandler.alloc().initWithData_options_(NSData.dataWithBytes_length_(jpeg, len(jpeg)), None)
        req = VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(1)  # accurate
        ok, err = handler.performRequests_error_([req], None)
        if not ok:
            log.debug("OCR failed: %s", err)
            return ""
        lines: list[str] = []
        for obs in req.results() or []:
            cand = obs.topCandidates_(1)
            if cand:
                lines.append(str(cand[0].string()))
        return "\n".join(lines)
    except Exception as exc:
        log.debug("OCR unavailable: %s", exc)
        return ""
