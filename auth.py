"""
auth.py — Kimlik Doğrulama ve Yetkilendirme Katmanı

Bu modül iki ayrı token tipini doğrular:

  1. KULLANICI TOKEN'I  (verify_user_context)
     Bilimp'in, soru soran son kullanıcı adına ürettiği kısa ömürlü token.
     Yalnızca POST /api/v1/query ucunda kabul edilir. İçinde kullanıcının
     erişim özniteliklerini taşıyan 'user_context' claim'i bulunur.

  2. SERVİS TOKEN'I  (verify_service_token)
     Yalnızca Bilimp'in sunucu tarafına (arka uca) ait, herhangi bir son
     kullanıcıyla ilişkilendirilmemiş token. Hedef kitle yönetimi uçlarında
     (PUT/GET /audience, GET /audience-compliance-report) kabul edilir.

     Bu ayrım kasıtlıdır: hedef kitle değişikliği güvenlik açısından hassas
     bir işlemdir ve yalnızca Bilimp'in kendi yetki kontrolünden geçmiş bir
     istek tarafından tetiklenmelidir. Bu iki uç son kullanıcı token'ını
     kabul etseydi, Bilimp arayüzünü hiç kullanmadan doğrudan API'ye istek
     atan herhangi bir kullanıcı, kendi sorgu token'ıyla herhangi bir
     dokümanın hedef kitlesini değiştirebilirdi.

Her iki token tipi de aynı imzalama altyapısını (RS256, çoklu anahtar/kid
desteği) paylaşır, yalnızca 'aud' (audience) claim'i ile ayrışır.

────────────────────────────────────────────────────────────────────────────
Beklenen JWT Yapısı

Kullanıcı token'ı:
    {
      "iss": "<JWT_ISSUER>",
      "aud": "<JWT_AUDIENCE_QUERY>",
      "sub": "613",                    # user_context.kullanici_id ile tutarlı olmalı
      "iat": <unix ts>, "exp": <unix ts>,
      "user_context": { "sirket_id": 14, ..., "kullanici_id": 613 }
    }

Servis token'ı:
    {
      "iss": "<JWT_ISSUER>",
      "aud": "<JWT_AUDIENCE_ADMIN>",
      "sub": "bilimp-backend",
      "iat": <unix ts>, "exp": <unix ts>
    }

────────────────────────────────────────────────────────────────────────────
Ortam Değişkenleri

  JWT_ALGORITHM             — varsayılan "RS256"
  JWT_ISSUER                — beklenen 'iss' claim değeri (ZORUNLU)
  JWT_AUDIENCE_QUERY         — kullanıcı token'ları için beklenen 'aud' (ZORUNLU)
  JWT_AUDIENCE_ADMIN         — servis token'ları için beklenen 'aud' (ZORUNLU)
  JWT_CLOCK_SKEW_SECONDS     — iat/exp doğrulamasında tolerans, varsayılan 30
  JWT_PUBLIC_KEYS_JSON       — {"kid1": "-----BEGIN PUBLIC KEY-----...", ...}
                               Anahtar rotasyonu için çoklu genel anahtar haritası.
  JWT_PUBLIC_KEY             — Tek anahtarlı basit kurulum için PEM (kid yoksa
                               veya JWT_PUBLIC_KEYS_JSON'da eşleşme bulunamazsa
                               yedek olarak kullanılır).

Not: Bu servis yalnızca GENEL anahtarı tutar. İmzalama için kullanılan ÖZEL
anahtar yalnızca token'ı üreten tarafta (Bilimp) kalır; bu, RS256'nın HS256'ya
göre temel avantajıdır — anahtar sızsa bile üçüncü bir taraf token üretemez.
"""

from __future__ import annotations
import os
import json
import logging
from datetime import timedelta
from typing import Optional

import jwt  # PyJWT
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from abac import UserContext

logger = logging.getLogger("auth")

_bearer = HTTPBearer(auto_error=True)


# ══════════════════════════════════════════════════════════════════════════════
# Yapılandırma ve Anahtar Çözümleme
# ══════════════════════════════════════════════════════════════════════════════
def _algorithm() -> str:
    return os.getenv("JWT_ALGORITHM", "RS256")


def _issuer() -> str:
    issuer = os.getenv("JWT_ISSUER")
    if not issuer:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sunucu kimlik doğrulama yapılandırması eksik (JWT_ISSUER).",
        )
    return issuer


def _clock_skew() -> timedelta:
    try:
        seconds = int(os.getenv("JWT_CLOCK_SKEW_SECONDS", "30"))
    except ValueError:
        seconds = 30
    return timedelta(seconds=seconds)


def _normalize_pem(value: str) -> str:
    """
    .env dosyalarında PEM anahtarları tek satıra sığdırmak için genelde
    gerçek satır sonu yerine kaçış dizisi '\\n' kullanılır (Docker Compose'un
    env_file okuyucusu çok satırlı değerleri güvenilir işlemez). Bu fonksiyon
    '\\n' dizisini gerçek satır sonuna çevirir; değer zaten gerçek satır
    sonlarıyla geldiyse (yerel .env / python-dotenv testinde olduğu gibi)
    hiçbir şeyi bozmadan olduğu gibi bırakır.
    """
    return value.replace("\\n", "\n") if value else value


def _load_public_keys() -> dict[str, str]:
    """kid → PEM genel anahtar haritasını döner. Rotasyon desteği için."""
    raw = os.getenv("JWT_PUBLIC_KEYS_JSON")
    if not raw:
        return {}
    try:
        keys = json.loads(raw)
        if not isinstance(keys, dict):
            raise ValueError("JWT_PUBLIC_KEYS_JSON bir JSON nesnesi olmalı.")
        return {kid: _normalize_pem(pem) for kid, pem in keys.items()}
    except (json.JSONDecodeError, ValueError):
        logger.error("JWT_PUBLIC_KEYS_JSON çözümlenemedi; yapılandırma hatalı.")
        return {}


def _resolve_public_key(token: str) -> str:
    """
    Token header'ındaki 'kid' alanına göre doğru genel anahtarı seçer.
    'kid' yoksa veya haritada bulunamazsa JWT_PUBLIC_KEY yedeğine düşer.

    İki farklı hata durumu kasıtlı olarak ayrıştırılır:
      - Sunucuda HİÇBİR anahtar yapılandırılmamışsa → 500 (gerçek yapılandırma
        hatası; bu bizim hatamız, saldırganın değil).
      - En az bir anahtar yapılandırılmış ama token'ın 'kid'i hiçbiriyle
        eşleşmiyorsa → 401 (bu sunucu hatası değil, güvenilmeyen/geçersiz bir
        token'dır — 500 dönmek saldırgana "sunucu yanlış yapılandırılmış"
        bilgisini sızdırır ve izlemede gerçek arıza gibi yanlış alarm üretir).
    """
    keys = _load_public_keys()
    fallback = _normalize_pem(os.getenv("JWT_PUBLIC_KEY"))

    if not keys and not fallback:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sunucu kimlik doğrulama yapılandırması eksik (genel anahtar bulunamadı).",
        )

    kid: Optional[str] = None
    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
    except jwt.InvalidTokenError:
        pass  # Header okunamazsa aşağıdaki decode() zaten 401 üretecek.

    if kid and kid in keys:
        return keys[kid]
    if fallback:
        return fallback

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Geçersiz kimlik token'ı (tanınmayan anahtar kimliği).",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Ortak Token Doğrulama
# ══════════════════════════════════════════════════════════════════════════════
def _decode_and_verify(token: str, expected_audience: str) -> dict:
    """
    İmza, iss, aud, iat/exp (clock skew toleranslı) doğrulamasını yapar.
    Başarılıysa çözümlenmiş payload'ı döner. Aksi halde 401 fırlatır.
    """
    public_key = _resolve_public_key(token)

    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=[_algorithm()],
            issuer=_issuer(),
            audience=expected_audience,
            leeway=_clock_skew(),
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token süresi dolmuş.")
    except jwt.InvalidIssuerError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token 'iss' claim'i geçersiz.")
    except jwt.InvalidAudienceError:
        # (Faz 4 / madde 15) 401 değil 403: imza/iss geçerli — token GERÇEKTEN
        # Bilimp tarafından üretilmiş ve güvenilir. Sorun kimliğin kendisi
        # değil, bu token'ın BU uç için yetkili olmaması (yanlış tür: kullanıcı
        # token'ı ile servis ucu ya da tersi). "Kimliğiniz doğrulanamadı"
        # (401) ile "kimliğiniz doğru ama bu işlem için yetkiniz yok" (403)
        # farklı durumlardır ve istemci tarafında farklı ele alınmalıdır.
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Bu token bu uç nokta için yetkili değil (aud uyuşmazlığı). "
            "Kullanıcı token'ı ile servis-yalnızca uçlar çağrılamaz ve tersi.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Geçersiz kimlik token'ı.")

    return payload


# ══════════════════════════════════════════════════════════════════════════════
# 1) Kullanıcı Token'ı — POST /api/v1/query
# ══════════════════════════════════════════════════════════════════════════════
def verify_user_context(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> UserContext:
    """
    Yalnızca sorgu ucunda kullanılır. Token'ın 'aud' claim'i JWT_AUDIENCE_QUERY
    ile eşleşmelidir; aksi halde (örneğin bir servis token'ı buraya gelirse)
    401 döner.

    sub / user_context.kullanici_id çelişki kuralı: her ikisi de mevcutsa ve
    sayısal olarak birbirini doğrulamıyorsa istek REDDEDİLİR (fail-closed).
    Bu, iki claim'den birinin sonradan/hatalı üretilmiş olabileceği bir
    token'a güvenmemek içindir.
    """
    audience = os.getenv("JWT_AUDIENCE_QUERY")
    if not audience:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Sunucu kimlik doğrulama yapılandırması eksik (JWT_AUDIENCE_QUERY).",
        )

    payload = _decode_and_verify(credentials.credentials, audience)

    uc_claim = payload.get("user_context")
    if not isinstance(uc_claim, dict):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Token içinde 'user_context' claim'i bulunamadı.",
        )

    try:
        user = UserContext(**uc_claim)
    except Exception:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Kullanıcı bağlamı (user_context) geçersiz biçimde.",
        )

    sub = payload.get("sub")
    if sub is not None and user.kullanici_id is not None:
        try:
            if int(sub) != int(user.kullanici_id):
                raise HTTPException(
                    status.HTTP_401_UNAUTHORIZED,
                    "Token tutarsız: 'sub' ve 'user_context.kullanici_id' "
                    "aynı kullanıcıyı işaret etmiyor.",
                )
        except (TypeError, ValueError):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Token tutarsız: 'sub' claim'i sayısal bir kullanıcı kimliğine "
                "dönüştürülemedi.",
            )

    return user


# ══════════════════════════════════════════════════════════════════════════════
# 2) Servis Token'ı — Hedef Kitle Yönetimi Uçları
# ══════════════════════════════════════════════════════════════════════════════
class ServiceIdentity:
    """Servis token'ının doğrulanmasından dönen minimal kimlik bilgisi."""

    def __init__(self, subject: str):
        self.subject = subject

    def __repr__(self) -> str:
        return f"ServiceIdentity(subject={self.subject!r})"


def verify_service_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> ServiceIdentity:
    """
    Yalnızca PUT/GET /api/v1/documents/{id}/audience ve
    GET /api/v1/documents/audience-compliance-report uçlarında kullanılır.

    Bu uçlar bir son kullanıcı token'ıyla ASLA çağrılamaz — token'ın 'aud'
    claim'i JWT_AUDIENCE_ADMIN olmalıdır. Bu sayede hedef kitle değişikliği
    yalnızca Bilimp'in kendi yetki kontrolünden geçmiş sunucu-taraflı bir
    istekle tetiklenebilir; bireysel bir kullanıcının sorgu token'ıyla bu
    uçlara doğrudan erişimi mümkün değildir.
    """
    audience = os.getenv("JWT_AUDIENCE_ADMIN")
    if not audience:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Sunucu kimlik doğrulama yapılandırması eksik (JWT_AUDIENCE_ADMIN).",
        )

    payload = _decode_and_verify(credentials.credentials, audience)
    subject = payload.get("sub") or "bilinmeyen-servis"
    return ServiceIdentity(subject=subject)