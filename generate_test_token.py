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


# Test için kullanılan sabit müşteri kimliği — kullanıcı ve servis token'ı
# AYNI müşteriye ait olmalı ki testlerde birbirleriyle tutarlı çalışsınlar.
MUSTERI_ID = 501

USER_CONTEXT = {
    "musteri_id": MUSTERI_ID,
    "sirket_ids": [14, 99],
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
        "musteri_id": MUSTERI_ID,
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }
    return jwt.encode(payload, _private_key(), algorithm="RS256")


def main():
    kind = (sys.argv[1] if len(sys.argv) > 1 else "user").lower()

    if kind == "service":
        token = make_service_token()
        print(f"\n=== SERVİS TOKEN'I (müşteri {MUSTERI_ID} için, 1 saat geçerli) ===\n")
        print(token)
        print("\nNot: musteri_id artık İSTEK PARAMETRESİ DEĞİL — bu token'ın içinde")
        print(f"     taşınıyor (musteri_id={MUSTERI_ID}). Yönetim uçlarında ayrıca")
        print("     sirket_id/musteri_id belirtmenize gerek yok.\n")
        print("=== Örnek: Doküman Oluştur (POST /documents) ===\n")
        print("curl -X POST http://localhost:8000/api/v1/documents \\")
        print(f'  -H "Authorization: Bearer {token[:24]}..." \\')
        print('  -H "Content-Type: application/json" \\')
        print("  -d '{\"dokuman_id\": \"test.txt\", \"dosya_icerigi_base64\": \"<base64>\", "
              "\"audience_policy\": {\"rules\": [{\"sirket_ids\": [14]}]}, "
              "\"yukleyen_kullanici_id\": 42}'")
        print("\n=== Örnek: Hedef Kitle Güncelle (PUT /audience) ===\n")
        print("curl -X PUT http://localhost:8000/api/v1/documents/test.txt/audience \\")
        print(f'  -H "Authorization: Bearer {token[:24]}..." \\')
        print('  -H "Content-Type: application/json" \\')
        print("  -d '{\"audience_policy\": {\"rules\": [{\"bina_ids\": [16]}]}, "
              "\"beklenen_audience_versiyon\": 1, \"degistiren_kullanici_id\": 42}'")
        print("\n=== Örnek: Doküman Sil (DELETE) — artık gövde gerektirir ===\n")
        print("curl -X DELETE http://localhost:8000/api/v1/documents/test.txt \\")
        print(f'  -H "Authorization: Bearer {token[:24]}..." \\')
        print('  -H "Content-Type: application/json" \\')
        print("  -d '{\"beklenen_versiyon\": 1, \"degistiren_kullanici_id\": 42}'")
    else:
        token = make_user_token()
        print("\n=== KULLANICI TOKEN'I (/api/v1/query için, 1 saat geçerli) ===\n")
        print(token)
        print("\n=== İçindeki UserContext ===\n")
        print(json.dumps(USER_CONTEXT, indent=2, ensure_ascii=False))
        print("\n=== Örnek curl ===\n")
        print("curl -X POST http://localhost:8000/api/v1/query \\")
        print(f'  -H "Authorization: Bearer {token[:24]}..." \\')
        print('  -H "Content-Type: application/json" \\')
        print("  -d '{\"soru\": \"İş sağlığı eğitimi ne zaman?\"}'")
    print()


if __name__ == "__main__":
    main()