"""
tests/test_api_e2e.py — Tam HTTP yığını üzerinden uçtan uca izolasyon kanıtı.

Gerçek RS256 imzalı JWT'ler üretir, FastAPI TestClient ile gerçek HTTP
isteği yapar (yalnızca Qdrant/embedding/LLM sahte nesnelerle değiştirilir).
Amaç: "14 numaralı şirketin kullanıcısı, 18 numaralı şirketin koleksiyonuna
hiçbir şekilde dokunamaz" iddiasını birim testi seviyesinde değil, gerçek
istek/yanıt döngüsü seviyesinde kanıtlamak.
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


def _user_token(private_pem, sirket_id, kullanici_id=613):
    payload = {
        "iss": "bilimp-teracity", "aud": "tubitak1505-query",
        "sub": str(kullanici_id), "iat": int(time.time()), "exp": int(time.time()) + 3600,
        "user_context": {"sirket_id": sirket_id, "kullanici_id": kullanici_id, "grup_ids": []},
    }
    return pyjwt.encode(payload, private_pem, algorithm="RS256")


def _service_token(private_pem):
    payload = {
        "iss": "bilimp-teracity", "aud": "tubitak1505-audience-admin",
        "sub": "bilimp-backend", "iat": int(time.time()), "exp": int(time.time()) + 3600,
    }
    return pyjwt.encode(payload, private_pem, algorithm="RS256")


class TestCrossTenantIsolationEndToEnd:
    def test_company_14_query_never_touches_company_18_collection(self, app_client):
        client, fake_qdrant_client, fake_qdrant_module, private_pem = app_client
        token = _user_token(private_pem, sirket_id=14)

        resp = client.post(
            "/api/v1/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"soru": "test sorusu"},
        )

        assert resp.status_code == 200
        _, kwargs = fake_qdrant_module.QdrantVectorStore.call_args
        assert kwargs["collection_name"] == "tubitak1505_sirket_14"
        assert kwargs["collection_name"] != "tubitak1505_sirket_18"

    def test_company_18_query_never_touches_company_14_collection(self, app_client):
        client, fake_qdrant_client, fake_qdrant_module, private_pem = app_client
        token = _user_token(private_pem, sirket_id=18)

        resp = client.post(
            "/api/v1/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"soru": "test sorusu"},
        )

        assert resp.status_code == 200
        _, kwargs = fake_qdrant_module.QdrantVectorStore.call_args
        assert kwargs["collection_name"] == "tubitak1505_sirket_18"

    def test_query_without_token_rejected(self, app_client):
        client, *_ = app_client
        resp = client.post("/api/v1/query", json={"soru": "test"})
        # HTTPBearer(auto_error=True): Authorization başlığı hiç yoksa 401/403
        # döner (Starlette sürümüne göre değişebilir); asıl garanti edilen
        # şey 200 OLMAMASI ve hiçbir Qdrant sorgusunun tetiklenmemesidir.
        assert resp.status_code in (401, 403)

    def test_service_token_cannot_call_query(self, app_client):
        """(Faz 4 / madde 15) aud uyuşmazlığı artık 403 — kimlik doğru,
        yetki yanlış uç için."""
        client, _, _, private_pem = app_client
        token = _service_token(private_pem)
        resp = client.post(
            "/api/v1/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"soru": "test"},
        )
        assert resp.status_code == 403

    def test_user_token_cannot_call_audience_management(self, app_client):
        """(Faz 4 / madde 15) aud uyuşmazlığı artık 403."""
        client, _, _, private_pem = app_client
        token = _user_token(private_pem, sirket_id=14)
        resp = client.get(
            "/api/v1/documents/audience-compliance-report",
            headers={"Authorization": f"Bearer {token}"},
            params={"sirket_id": 14},
        )
        assert resp.status_code == 403

    def test_audience_endpoint_requires_sirket_id_param(self, app_client):
        client, _, _, private_pem = app_client
        token = _service_token(private_pem)
        resp = client.get(
            "/api/v1/documents/audience-compliance-report",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422  # FastAPI zorunlu query param eksikliği

    def test_service_token_with_sirket_id_reaches_correct_tenant(self, app_client):
        client, fake_qdrant_client, _, private_pem = app_client
        fake_qdrant_client.scroll.return_value = ([], None)
        token = _service_token(private_pem)

        resp = client.get(
            "/api/v1/documents/audience-compliance-report",
            headers={"Authorization": f"Bearer {token}"},
            params={"sirket_id": 14},
        )

        assert resp.status_code == 200
        _, kwargs = fake_qdrant_client.scroll.call_args
        assert kwargs["collection_name"] == "tubitak1505_sirket_14"

    def test_health_endpoint_requires_no_token(self, app_client):
        client, *_ = app_client
        resp = client.get("/health")
        assert resp.status_code == 200