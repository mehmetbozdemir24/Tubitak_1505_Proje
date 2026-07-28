"""
rate_limiting.py — İstek Sınırlama

Bu modül, tek bir API örneği (instance) için bellek-içi bir sabit-pencere
(fixed-window) sınırlayıcı uygular.

ÖNEMLİ ÖLÇEKLENDİRME NOTU: Bu uygulama BELLEK-İÇİDİR — birden fazla API
örneği (yatay ölçekleme / birden fazla container) arkasında çalışırsa, her
örnek kendi sayacını tutar ve gerçek toplam sınır bu örnek sayısıyla çarpılır.
Çok-örnekli bir dağıtımda paylaşılan bir sayaç deposu (Redis vb.) gerekir.
Şu anki tek-örnekli dağıtım için bu yeterlidir; ölçek büyüdüğünde bu modülün
Redis-destekli bir sürümle değiştirilmesi önerilir.

TASARIM KARARI (v2.0) — Anahtar müşteri (musteri_id) + kullanıcı (sub) çifti:
Yalnızca 'sub' (kullanıcı kimliği) ile anahtarlamak yeterli DEĞİLDİR: Bilimp
çok müşterili bir üründür ve her müşteri kendi veritabanında bağımsız
kullanıcı kimlikleri üretir — iki farklı müşterinin kullanıcı_id=613 olan
çalışanları rastlantısal olarak aynı 'sub' değerine sahip olabilir, bu da bir
müşterinin kullanıcısının, başka bir müşterinin kotasını (istemeden)
paylaşmasına yol açar. Bu yüzden anahtar (musteri_id, sub) ÇİFTİDİR.
"""

from __future__ import annotations
import os
import time
import uuid
import threading
from collections import defaultdict, deque

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class _FixedWindowLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            q = self._hits[key]
            while q and (now - q[0]) > self.window_seconds:
                q.popleft()
            if len(q) >= self.max_requests:
                return False
            q.append(now)
            return True


def _client_identity(request: Request) -> str:
    """
    Rate-limit kovası anahtarını üretir. Token'ın İMZASI BURADA
    DOĞRULANMAZ (yalnızca kova seçimi için okunur; gerçek yetkilendirme
    auth.py'nin doğrulanmış bağımlılıklarında ayrıca yapılır) — bu yüzden
    sahte bir 'musteri_id'/'sub' ile daha büyük bir kotaya "sıçramak"
    mümkün değildir, sadece kendi (doğrulanmamış) kovasına yazar; asıl
    işlem imzası geçersiz bir token'la zaten reddedilir.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:]
        try:
            import jwt as pyjwt
            unverified = pyjwt.decode(token, options={"verify_signature": False})
            sub = unverified.get("sub")
            # Kullanıcı token'ında musteri_id user_context içinde, servis
            # token'ında ise üst seviyededir (bkz. auth.py).
            musteri_id = unverified.get("musteri_id")
            if musteri_id is None:
                uc = unverified.get("user_context")
                if isinstance(uc, dict):
                    musteri_id = uc.get("musteri_id")
            if sub and musteri_id is not None:
                return f"musteri:{musteri_id}:sub:{sub}"
            if sub:
                return f"sub:{sub}"
        except Exception:
            pass
    client_host = request.client.host if request.client else "bilinmeyen"
    return f"ip:{client_host}"


def install_rate_limiting(app: FastAPI) -> None:
    """
    api.py'nin FastAPI app nesnesine rate-limit middleware'ini kaydeder.

    Ortam değişkenleri:
      RATE_LIMIT_REQUESTS         — pencere başına izin verilen istek sayısı (varsayılan 60)
      RATE_LIMIT_WINDOW_SECONDS   — pencere genişliği, saniye (varsayılan 60)
    """
    max_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
    window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    limiter = _FixedWindowLimiter(max_requests, window_seconds)

    @app.middleware("http")
    async def _rate_limit_middleware(request: Request, call_next):
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        key = _client_identity(request)
        if not limiter.allow(key):
            # ÖNEMLİ: Starlette'in @app.middleware("http") katmanı, burada
            # fırlatılan bir HTTPException'ı app.exception_handler(HTTPException)
            # ile DOĞRU eşleştirmez (middleware, route/exception-handler
            # zincirinin DIŞINDA çalışır) — bu yüzden JSONResponse doğrudan
            # burada, error_handling.py'deki şemayla BİREBİR aynı biçimde
            # döndürülür.
            trace_id = str(uuid.uuid4())
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "kod": "COK_FAZLA_ISTEK",
                    "mesaj": (
                        f"İstek sınırı aşıldı ({max_requests} istek / "
                        f"{window_seconds} sn). Lütfen bir süre sonra tekrar deneyin."
                    ),
                    "trace_id": trace_id,
                },
                headers={"X-Trace-Id": trace_id},
            )
        return await call_next(request)