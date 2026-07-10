"""
error_handling.py — Standart Hata Gövdesi ve İzlenebilirlik (Faz 4 / madde 15)

Teracity'nin bulgusu: "Hata yanıtlarının gövde şeması (kod, mesaj, trace id)
tanımlı değil." Bu modül, TÜM hata yanıtlarının (FastAPI'nin varsayılan
{"detail": "..."} biçimi yerine) tutarlı bir şema kullanmasını sağlar:

    { "kod": "...", "mesaj": "...", "trace_id": "..." }

trace_id, her isteğe atanan benzersiz bir tanımlayıcıdır ve HEM yanıtta HEM
sunucu loglarında bulunur — bu sayede Teracity bir hatayı bildirdiğinde,
trace_id'yi paylaşarak sunucu tarafındaki tam kaydı (stack trace dahil)
saniyeler içinde bulmak mümkün olur; aksi halde "hangi istekti bu"
sorusuna zaman zaman kesin cevap verilemez.

Kullanım (api.py içinde):
    from error_handling import install_error_handling
    install_error_handling(app)
"""

from __future__ import annotations
import logging
import uuid

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("error_handling")


class ErrorResponse(BaseModel):
    kod: str
    mesaj: str
    trace_id: str


_STATUS_TO_KOD = {
    status.HTTP_400_BAD_REQUEST: "GECERSIZ_ISTEK",
    status.HTTP_401_UNAUTHORIZED: "YETKISIZ",
    status.HTTP_403_FORBIDDEN: "YASAK",
    status.HTTP_404_NOT_FOUND: "BULUNAMADI",
    status.HTTP_409_CONFLICT: "CAKISMA",
    status.HTTP_413_CONTENT_TOO_LARGE: "COK_BUYUK",
    status.HTTP_422_UNPROCESSABLE_CONTENT: "GECERSIZ_ICERIK",
    status.HTTP_429_TOO_MANY_REQUESTS: "COK_FAZLA_ISTEK",
    status.HTTP_500_INTERNAL_SERVER_ERROR: "SUNUCU_HATASI",
    status.HTTP_503_SERVICE_UNAVAILABLE: "SERVIS_KULLANILAMAZ",
}


def _kod_for_status(status_code: int) -> str:
    return _STATUS_TO_KOD.get(status_code, "BILINMEYEN_HATA")


def install_error_handling(app: FastAPI) -> None:
    """
    api.py'nin FastAPI app nesnesine, standart hata şemasını ve trace_id
    üretimini kaydeder. Uygulama başlatılırken bir kez çağrılmalıdır.
    """

    @app.middleware("http")
    async def _trace_id_middleware(request: Request, call_next):
        trace_id = str(uuid.uuid4())
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException):
        trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
        logger.warning(
            "http_hata trace_id=%s durum=%d mesaj=%s yol=%s",
            trace_id, exc.status_code, exc.detail, request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                kod=_kod_for_status(exc.status_code),
                mesaj=str(exc.detail),
                trace_id=trace_id,
            ).model_dump(),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
        logger.exception("islenmemis_hata trace_id=%s yol=%s", trace_id, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                kod="SUNUCU_HATASI",
                mesaj="Beklenmeyen bir sunucu hatası oluştu. Sorun devam ederse "
                      "trace_id ile birlikte bildirin.",
                trace_id=trace_id,
            ).model_dump(),
        )