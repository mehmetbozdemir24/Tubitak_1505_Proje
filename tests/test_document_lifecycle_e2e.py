"""
tests/test_document_lifecycle_e2e.py — Doküman yaşam döngüsü uçlarının tam
HTTP yığını üzerinden testleri. test_api_e2e.py'deki fixture desenini
izler; ayrıca aynı dokuman_id'nin FARKLI tenant'larda çakışmadığını
(fiziksel izolasyonun doğal bir sonucu) kanıtlar.
"""
import sys
import time
import base64
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
    fake_qdrant_module.QdrantVectorStore.return_value = fake_store
    fake_qdrant_module.RetrievalMode.HYBRID = "hybrid"
    monkeypatch.setitem(sys.modules, "langchain_qdrant", fake_qdrant_module)
    monkeypatch.setitem(sys.modules, "language_utils", MagicMock())
    monkeypatch.setitem(sys.modules, "langchain_core.messages", MagicMock())
    monkeypatch.setitem(sys.modules, "pymupdf4llm", MagicMock())
    monkeypatch.setitem(sys.modules, "langchain_text_splitters", MagicMock())

    fake_chunker = MagicMock()
    from langchain_core.documents import Document
    fake_chunker.chunk_text.return_value = [Document(page_content="içerik", metadata={})]
    fake_chunker.chunk_pptx.return_value = [Document(page_content="içerik", metadata={})]
    monkeypatch.setitem(sys.modules, "chunker", fake_chunker)

    for mod in ["api", "rag_service", "document_ingestion_service"]:
        sys.modules.pop(mod, None)

    import api
    from fastapi.testclient import TestClient

    fake_client = MagicMock()
    fake_client.collection_exists.return_value = True
    api._resources["client"] = fake_client
    api._resources["dense"] = MagicMock()
    api._resources["sparse"] = MagicMock()
    api._resources["llm"] = MagicMock()

    return TestClient(api.app), fake_client, private_pem


def _service_token(private_pem):
    payload = {
        "iss": "bilimp-teracity", "aud": "tubitak1505-audience-admin",
        "sub": "bilimp-backend", "iat": int(time.time()), "exp": int(time.time()) + 3600,
    }
    return pyjwt.encode(payload, private_pem, algorithm="RS256")


class TestCreateDocumentEndpoint:
    def test_successful_creation_returns_201(self, app_client):
        client, fake_qdrant_client, private_pem = app_client
        fake_qdrant_client.collection_exists.return_value = False
        fake_qdrant_client.scroll.return_value = ([], None)
        token = _service_token(private_pem)

        resp = client.post(
            "/api/v1/documents",
            headers={"Authorization": f"Bearer {token}"},
            params={"sirket_id": 14},
            json={
                "dokuman_id": "rapor.pdf",
                "dosya_icerigi_base64": base64.b64encode(b"sahte icerik").decode(),
                "audience_policy": {"rules": [{"sirket_ids": [14]}]},
                "yukleyen_kullanici_id": 42,
            },
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["dokuman_id"] == "rapor.pdf"
        assert body["versiyon"] == 1

    def test_duplicate_dokuman_id_returns_409(self, app_client):
        client, fake_qdrant_client, private_pem = app_client
        fake_qdrant_client.collection_exists.return_value = True
        fake_qdrant_client.scroll.return_value = ([MagicMock()], None)
        token = _service_token(private_pem)

        resp = client.post(
            "/api/v1/documents",
            headers={"Authorization": f"Bearer {token}"},
            params={"sirket_id": 14},
            json={
                "dokuman_id": "rapor.pdf",
                "dosya_icerigi_base64": base64.b64encode(b"icerik").decode(),
                "audience_policy": {"rules": []},
                "yukleyen_kullanici_id": 42,
            },
        )
        assert resp.status_code == 409

    def test_invalid_base64_returns_422(self, app_client):
        client, _, private_pem = app_client
        token = _service_token(private_pem)

        resp = client.post(
            "/api/v1/documents",
            headers={"Authorization": f"Bearer {token}"},
            params={"sirket_id": 14},
            json={
                "dokuman_id": "rapor.pdf",
                "dosya_icerigi_base64": "!!!gecersiz!!!",
                "audience_policy": {"rules": []},
                "yukleyen_kullanici_id": 42,
            },
        )
        assert resp.status_code == 422

    def test_same_dokuman_id_different_tenants_does_not_conflict(self, app_client):
        client, fake_qdrant_client, private_pem = app_client
        fake_qdrant_client.collection_exists.return_value = False
        fake_qdrant_client.scroll.return_value = ([], None)
        token = _service_token(private_pem)

        payload = {
            "dokuman_id": "ortak_isimli_belge.pdf",
            "dosya_icerigi_base64": base64.b64encode(b"icerik").decode(),
            "audience_policy": {"rules": []},
            "yukleyen_kullanici_id": 42,
        }

        resp1 = client.post("/api/v1/documents", headers={"Authorization": f"Bearer {token}"},
                             params={"sirket_id": 14}, json=payload)
        resp2 = client.post("/api/v1/documents", headers={"Authorization": f"Bearer {token}"},
                             params={"sirket_id": 18}, json=payload)

        assert resp1.status_code == 201
        assert resp2.status_code == 201


class TestUpdateContentEndpoint:
    def test_version_conflict_returns_409(self, app_client):
        client, fake_qdrant_client, private_pem = app_client
        fake_qdrant_client.collection_exists.return_value = True
        existing = MagicMock()
        existing.payload = {"metadata": {"versiyon": 5, "audience": {"rules": []}}}
        fake_qdrant_client.scroll.return_value = ([existing], None)
        token = _service_token(private_pem)

        resp = client.put(
            "/api/v1/documents/rapor.pdf/content",
            headers={"Authorization": f"Bearer {token}"},
            params={"sirket_id": 14},
            json={
                "dosya_icerigi_base64": base64.b64encode(b"yeni icerik").decode(),
                "beklenen_versiyon": 1,
                "degistiren_kullanici_id": 42,
            },
        )
        assert resp.status_code == 409

    def test_not_found_returns_404(self, app_client):
        client, fake_qdrant_client, private_pem = app_client
        fake_qdrant_client.collection_exists.return_value = False
        token = _service_token(private_pem)

        resp = client.put(
            "/api/v1/documents/yok.pdf/content",
            headers={"Authorization": f"Bearer {token}"},
            params={"sirket_id": 14},
            json={
                "dosya_icerigi_base64": base64.b64encode(b"icerik").decode(),
                "beklenen_versiyon": 1,
                "degistiren_kullanici_id": 42,
            },
        )
        assert resp.status_code == 404


class TestDeleteEndpoint:
    def test_successful_delete(self, app_client):
        client, fake_qdrant_client, private_pem = app_client
        fake_qdrant_client.collection_exists.return_value = True
        fake_qdrant_client.scroll.return_value = ([MagicMock(id="a"), MagicMock(id="b")], None)
        token = _service_token(private_pem)

        resp = client.delete(
            "/api/v1/documents/rapor.pdf",
            headers={"Authorization": f"Bearer {token}"},
            params={"sirket_id": 14},
        )
        assert resp.status_code == 200
        assert resp.json()["silinen_nokta_sayisi"] == 2

    def test_not_found_returns_404(self, app_client):
        client, fake_qdrant_client, private_pem = app_client
        fake_qdrant_client.collection_exists.return_value = False
        token = _service_token(private_pem)

        resp = client.delete(
            "/api/v1/documents/yok.pdf",
            headers={"Authorization": f"Bearer {token}"},
            params={"sirket_id": 14},
        )
        assert resp.status_code == 404


class TestDocumentLifecycleRequiresServiceToken:
    def test_user_token_cannot_create_document(self, app_client):
        """(Faz 4 / madde 15) aud uyuşmazlığı artık 403."""
        client, _, private_pem = app_client
        user_token = pyjwt.encode({
            "iss": "bilimp-teracity", "aud": "tubitak1505-query",
            "sub": "613", "iat": int(time.time()), "exp": int(time.time()) + 3600,
            "user_context": {"sirket_id": 14, "kullanici_id": 613, "grup_ids": []},
        }, private_pem, algorithm="RS256")

        resp = client.post(
            "/api/v1/documents",
            headers={"Authorization": f"Bearer {user_token}"},
            params={"sirket_id": 14},
            json={
                "dokuman_id": "rapor.pdf",
                "dosya_icerigi_base64": base64.b64encode(b"icerik").decode(),
                "audience_policy": {"rules": []},
                "yukleyen_kullanici_id": 42,
            },
        )
        assert resp.status_code == 403