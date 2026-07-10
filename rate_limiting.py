"""
rate_limiting.py — İstek Sınırlama (Faz 4 / madde 15)

Teracity'nin bulgusu: "Rate limit ... belirtilmeli." Bu modül, tek bir API
örneği (instance) için bellek-içi bir sabit-pencere (fixed-window) sınırlayıcı
uygular.

ÖNEMLİ ÖLÇEKLENDİRME NOTU: Bu uygulama BELLEK-İÇİDİR — birden fazla API
örneği (yatay ölçekleme / birden fazla container) arkasında çalışırsa, her
örnek kendi sayacını tutar ve gerçek toplam sınır bu örnek sayısıyla çarpılır.
Çok-örnekli bir dağıtımda paylaşılan bir sayaç deposu (Redis vb.) gerekir.
Şu anki tek-örnekli dağıtım için bu yeterlidir; ölçek büyüdüğünde bu modülün
Redis-destekli bir sürümle değiştirilmesi önerilir.

Anahtar (rate limit kovası), kimliği doğrulanmış çağıran varsa onun 'sub'
claim'i, yoksa istemci IP adresidir — böylece bir kullanıcının/servisin
sınırı, aynı ağdaki diğer kullanıcıları etkilemez.
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
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:]
        try:
            import jwt as pyjwt
            unverified = pyjwt.decode(token, options={"verify_signature": False})
            sub = unverified.get("sub")
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