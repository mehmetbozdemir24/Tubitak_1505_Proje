"""
tests/test_api_e2e.py — Tam HTTP yığını üzerinden uçtan uca izolasyon kanıtı.

Gerçek RS256 imzalı JWT'ler üretir, FastAPI TestClient ile gerçek HTTP
isteği yapar (yalnızca Qdrant/embedding/LLM sahte nesnelerle değiştirilir).

(v2.0) Amaç: "501 numaralı MÜŞTERİNİN kullanıcısı, 777 numaralı MÜŞTERİNİN
koleksiyonuna hiçbir şekilde dokunamaz" iddiasını — hatta ikisi de AYNI
sirket_id'ye sahip olsa bile (Teracity'nin bulgusunun tam senaryosu) —
birim testi seviyesinde değil, gerçek istek/yanıt döngüsü seviyesinde
kanıtlamak.
"""
import sys
import time
from unittest.mock import MagicMock

import pytest
import jwt as pyjwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


@pytest.fixture(scope="module")
def keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


@pytest.fixture
def app_client(monkeypatch, keypair):
    private_pem, public_pem = keypair

    monkeypatch.setenv("JWT_ISSUER", "bilimp-teracity")
    monkeypatch.setenv("JWT_AUDIENCE_QUERY", "tubitak1505-query")
    monkeypatch.setenv("JWT_AUDIENCE_ADMIN", "tubitak1505-audience-admin")
    monkeypatch.setenv("JWT_PUBLIC_KEY", public_pem)

    fake_qdrant_module = MagicMock()
    fake_store = MagicMock()
    fake_store.similarity_search_with_score.return_value = []
    fake_qdrant_module.QdrantVectorStore.return_value = fake_store
    fake_qdrant_module.RetrievalMode.HYBRID = "hybrid"
    monkeypatch.setitem(sys.modules, "langchain_qdrant", fake_qdrant_module)
    monkeypatch.setitem(sys.modules, "language_utils", MagicMock())
    monkeypatch.setitem(sys.modules, "langchain_core.messages", MagicMock())

    for mod in ["api", "rag_service"]:
        sys.modules.pop(mod, None)

    import api
    from fastapi.testclient import TestClient

    fake_client = MagicMock()
    fake_client.collection_exists.return_value = True
    api._resources["client"] = fake_client
    api._resources["dense"] = MagicMock()
    api._resources["sparse"] = MagicMock()
    api._resources["llm"] = MagicMock()

    return TestClient(api.app), fake_client, fake_qdrant_module, private_pem


def _user_token(private_pem, musteri_id, sirket_ids=None, kullanici_id=613):
    payload = {
        "iss": "bilimp-teracity", "aud": "tubitak1505-query",
        "sub": str(kullanici_id), "iat": int(time.time()), "exp": int(time.time()) + 3600,
        "user_context": {
            "musteri_id": musteri_id,
            "sirket_ids": sirket_ids or [],
            "kullanici_id": kullanici_id, "grup_ids": [],
        },
    }
    return pyjwt.encode(payload, private_pem, algorithm="RS256")


def _service_token(private_pem, musteri_id):
    payload = {
        "iss": "bilimp-teracity", "aud": "tubitak1505-audience-admin",
        "sub": "bilimp-backend", "musteri_id": musteri_id,
        "iat": int(time.time()), "exp": int(time.time()) + 3600,
    }
    return pyjwt.encode(payload, private_pem, algorithm="RS256")


class TestCrossTenantIsolationEndToEnd:
    def test_customer_501_query_never_touches_customer_777_collection(self, app_client):
        client, fake_qdrant_client, fake_qdrant_module, private_pem = app_client
        token = _user_token(private_pem, musteri_id=501, sirket_ids=[14])

        resp = client.post(
            "/api/v1/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"soru": "test sorusu"},
        )

        assert resp.status_code == 200
        _, kwargs = fake_qdrant_module.QdrantVectorStore.call_args
        assert kwargs["collection_name"] == "tubitak1505_musteri_501"
        assert kwargs["collection_name"] != "tubitak1505_musteri_777"

    def test_customer_777_query_never_touches_customer_501_collection(self, app_client):
        client, fake_qdrant_client, fake_qdrant_module, private_pem = app_client
        token = _user_token(private_pem, musteri_id=777, sirket_ids=[14])

        resp = client.post(
            "/api/v1/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"soru": "test sorusu"},
        )

        assert resp.status_code == 200
        _, kwargs = fake_qdrant_module.QdrantVectorStore.call_args
        assert kwargs["collection_name"] == "tubitak1505_musteri_777"

    def test_exact_kullanici_id_collision_scenario_from_teracity_feedback(self, app_client):
        """
        Teracity'nin bulgusundaki BİREBİR örnek: "B müşterisinin
        kullanici_id=613 kullanıcısı, A müşterisi için yazılmış
        kullanici_ids:[613] kuralını sağlar." Bu senaryonun artık mümkün
        OLMADIĞINI kanıtlar — B müşterisinin sorgusu, A müşterisinin
        koleksiyonuna hiç ULAŞMAZ (fiziksel izolasyon), yani kullanici_id
        çakışması bir tehdit oluşturmadan önce sorgu zaten farklı bir
        koleksiyona yönlenmiş olur.
        """
        client, fake_qdrant_client, fake_qdrant_module, private_pem = app_client
        # A müşterisi (musteri_id=501) VE B müşterisi (musteri_id=999),
        # AYNI kullanici_id=613'e sahip — Teracity'nin senaryosu tam bu.
        token_musteri_a = _user_token(private_pem, musteri_id=501, kullanici_id=613)
        token_musteri_b = _user_token(private_pem, musteri_id=999, kullanici_id=613)

        client.post("/api/v1/query", headers={"Authorization": f"Bearer {token_musteri_a}"},
                    json={"soru": "test"})
        client.post("/api/v1/query", headers={"Authorization": f"Bearer {token_musteri_b}"},
                    json={"soru": "test"})

        collections_used = [
            c.kwargs["collection_name"] for c in fake_qdrant_module.QdrantVectorStore.call_args_list
        ]
        # İki sorgu da BAŞARILI oldu (kullanici_id çakışması hiçbir hataya
        # yol açmadı) ama FİZİKSEL OLARAK FARKLI koleksiyonlara gitti —
        # A'nın kullanici_ids:[613] kuralı B'nin koleksiyonuna hiç
        # uygulanma ihtimali bile olmadı.
        assert collections_used == ["tubitak1505_musteri_501", "tubitak1505_musteri_999"]

    def test_query_without_token_rejected(self, app_client):
        client, *_ = app_client
        resp = client.post("/api/v1/query", json={"soru": "test"})
        assert resp.status_code in (401, 403)

    def test_service_token_cannot_call_query(self, app_client):
        client, _, _, private_pem = app_client
        token = _service_token(private_pem, musteri_id=501)
        resp = client.post(
            "/api/v1/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"soru": "test"},
        )
        assert resp.status_code == 403

    def test_user_token_cannot_call_audience_management(self, app_client):
        client, _, _, private_pem = app_client
        token = _user_token(private_pem, musteri_id=501, sirket_ids=[14])
        resp = client.get(
            "/api/v1/documents/audience-compliance-report",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403

    def test_user_token_without_musteri_id_rejected_at_jwt_parsing(self, app_client):
        client, *_, private_pem = app_client
        payload = {
            "iss": "bilimp-teracity", "aud": "tubitak1505-query",
            "sub": "613", "iat": int(time.time()), "exp": int(time.time()) + 3600,
            "user_context": {"kullanici_id": 613, "grup_ids": []},
        }
        token = pyjwt.encode(payload, private_pem, algorithm="RS256")

        resp = client.post(
            "/api/v1/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"soru": "test"},
        )
        assert resp.status_code == 401

    def test_service_token_reaches_correct_customer_via_token_not_param(self, app_client):
        client, fake_qdrant_client, _, private_pem = app_client
        fake_qdrant_client.scroll.return_value = ([], None)
        token = _service_token(private_pem, musteri_id=501)

        resp = client.get(
            "/api/v1/documents/audience-compliance-report",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert resp.status_code == 200
        _, kwargs = fake_qdrant_client.scroll.call_args
        assert kwargs["collection_name"] == "tubitak1505_musteri_501"

    def test_health_endpoint_requires_no_token(self, app_client):
        client, *_ = app_client
        resp = client.get("/health")
        assert resp.status_code == 200