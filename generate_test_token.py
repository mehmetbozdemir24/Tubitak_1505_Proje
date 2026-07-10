from __future__ import annotations
import os
import sys
import time
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import jwt  # PyJWT


USER_CONTEXT = {
    "sirket_id": 14,
    "sube_id": None,
    "mudurluk_id": None,
    "birim_id": None,
    "grup_ids": [101, 108],
    "bina_id": 16,
    "pozisyon_id": None,
    "personel_tip_id": 1,
    "kullanici_id": 613,
    "yaka_tipi_id": None,
}


def _private_key() -> str:
    try:
        return open("private_key.pem").read()
    except FileNotFoundError:
        raise SystemExit(
            "HATA: private_key.pem bulunamadı. Önce çalıştırın:\n"
            "  python generate_test_keypair.py"
        )


def _issuer() -> str:
    return os.getenv("JWT_ISSUER", "bilimp-teracity")


def make_user_token() -> str:
    payload = {
        "iss": _issuer(),
        "aud": os.getenv("JWT_AUDIENCE_QUERY", "tubitak1505-query"),
        "sub": str(USER_CONTEXT["kullanici_id"]),
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
        "user_context": USER_CONTEXT,
    }
    return jwt.encode(payload, _private_key(), algorithm="RS256")


def make_service_token() -> str:
    payload = {
        "iss": _issuer(),
        "aud": os.getenv("JWT_AUDIENCE_ADMIN", "tubitak1505-audience-admin"),
        "sub": "bilimp-backend",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }
    return jwt.encode(payload, _private_key(), algorithm="RS256")


def main():
    kind = (sys.argv[1] if len(sys.argv) > 1 else "user").lower()

    if kind == "service":
        token = make_service_token()
        print("\n=== SERVİS TOKEN'I (hedef kitle uçları için, 1 saat geçerli) ===\n")
        print(token)
        print("\n=== Örnek curl (PUT /audience) ===\n")
        print(f'curl -X PUT http://localhost:8000/api/v1/documents/isg-egitimi-2026/audience \\')
        print(f'  -H "Authorization: Bearer {token[:24]}..." \\')
        print(f'  -H "Content-Type: application/json" \\')
        print('  -d \'{"audience_policy": {"rules": [{"bina_ids": [16]}]}, "degistiren_kullanici_id": 42}\'')
    else:
        token = make_user_token()
        print("\n=== KULLANICI TOKEN'I (/api/v1/query için, 1 saat geçerli) ===\n")
        print(token)
        print("\n=== İçindeki UserContext ===\n")
        print(json.dumps(USER_CONTEXT, indent=2, ensure_ascii=False))
        print("\n=== Örnek curl ===\n")
        print(f'curl -X POST http://localhost:8000/api/v1/query \\')
        print(f'  -H "Authorization: Bearer {token[:24]}..." \\')
        print(f'  -H "Content-Type: application/json" \\')
        print(f"  -d '{{\"soru\": \"İş sağlığı eğitimi ne zaman?\"}}'")
    print()


if __name__ == "__main__":
    main()