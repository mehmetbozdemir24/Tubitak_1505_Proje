"""
tests/test_document_lifecycle_e2e.py — Doküman yaşam döngüsü uçlarının tam
HTTP yığını üzerinden testleri. Ayrıca aynı dokuman_id'nin FARKLI
müşterilerde çakışmadığını (fiziksel izolasyonun doğal bir sonucu) kanıtlar.

(v2.0) musteri_id artık bir istek parametresi DEĞİLDİR — servis token'ının
kendisinde taşınır (bkz. auth.py). Bu yüzden "farklı tenant" senaryoları
artık farklı bir query param DEĞİL, farklı bir musteri_id'ye sahip AYRI bir
servis token'ı ile temsil edilir.
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

    # (v2.1) document_ingestion_service artık chunker.py değil vllm_ingestion.py
    # kullanıyor (Docling+VLM+contextual). Docling ağır/GPU-odaklı bir bağımlılık
    # olduğundan ve burada test edilen şey ayrıştırma değil HTTP+yaşam döngüsü
    # mantığı olduğundan, modülün TAMAMI sahteleriyle değiştirilir.
    fake_vllm_ingestion = MagicMock()
    from langchain_core.documents import Document
    fake_vllm_ingestion.ingest_file.return_value = [Document(page_content="içerik", metadata={})]
    fake_vllm_ingestion.ensure_ctx_model.return_value = None
    monkeypatch.setitem(sys.modules, "vllm_ingestion", fake_vllm_ingestion)

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


def _service_token(private_pem, musteri_id=501):
    payload = {
        "iss": "bilimp-teracity", "aud": "tubitak1505-audience-admin",
        "sub": "bilimp-backend", "musteri_id": musteri_id,
        "iat": int(time.time()), "exp": int(time.time()) + 3600,
    }
    return pyjwt.encode(payload, private_pem, algorithm="RS256")


class TestCreateDocumentEndpoint:
    def test_successful_creation_returns_201(self, app_client):
        client, fake_qdrant_client, private_pem = app_client
        fake_qdrant_client.collection_exists.return_value = True
        fake_qdrant_client.scroll.return_value = ([], None)
        token = _service_token(private_pem)

        resp = client.post(
            "/api/v1/documents",
            headers={"Authorization": f"Bearer {token}"},
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
            json={
                "dokuman_id": "rapor.pdf",
                "dosya_icerigi_base64": base64.b64encode(b"icerik").decode(),
                "audience_policy": {"rules": []},
                "allow_empty": True,
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
            json={
                "dokuman_id": "rapor.pdf",
                "dosya_icerigi_base64": "!!!gecersiz!!!",
                "audience_policy": {"rules": []},
                "yukleyen_kullanici_id": 42,
            },
        )
        assert resp.status_code == 422

    def test_same_dokuman_id_different_customers_does_not_conflict(self, app_client):
        client, fake_qdrant_client, private_pem = app_client
        fake_qdrant_client.collection_exists.return_value = True
        fake_qdrant_client.scroll.return_value = ([], None)
        token_musteri_501 = _service_token(private_pem, musteri_id=501)
        token_musteri_777 = _service_token(private_pem, musteri_id=777)

        payload = {
            "dokuman_id": "ortak_isimli_belge.pdf",
            "dosya_icerigi_base64": base64.b64encode(b"icerik").decode(),
            "audience_policy": {"rules": []},
            "allow_empty": True,
            "yukleyen_kullanici_id": 42,
        }

        resp1 = client.post("/api/v1/documents",
                             headers={"Authorization": f"Bearer {token_musteri_501}"}, json=payload)
        resp2 = client.post("/api/v1/documents",
                             headers={"Authorization": f"Bearer {token_musteri_777}"}, json=payload)

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
            json={
                "dosya_icerigi_base64": base64.b64encode(b"icerik").decode(),
                "beklenen_versiyon": 1,
                "degistiren_kullanici_id": 42,
            },
        )
        assert resp.status_code == 404


class TestDeleteEndpoint:
    def _fake_point(self, point_id, versiyon=1):
        p = MagicMock()
        p.id = point_id
        p.payload = {"metadata": {"versiyon": versiyon, "source": "rapor.pdf"}}
        return p

    def test_successful_delete(self, app_client):
        client, fake_qdrant_client, private_pem = app_client
        fake_qdrant_client.collection_exists.return_value = True
        fake_qdrant_client.scroll.return_value = (
            [self._fake_point("a", versiyon=1), self._fake_point("b", versiyon=1)], None
        )
        token = _service_token(private_pem)

        resp = client.request(
            "DELETE", "/api/v1/documents/rapor.pdf",
            headers={"Authorization": f"Bearer {token}"},
            json={"beklenen_versiyon": 1, "degistiren_kullanici_id": 42},
        )
        assert resp.status_code == 200
        assert resp.json()["silinen_nokta_sayisi"] == 2

    def test_version_conflict_returns_409(self, app_client):
        """(madde 5) DELETE artık optimistic locking'e tabi."""
        client, fake_qdrant_client, private_pem = app_client
        fake_qdrant_client.collection_exists.return_value = True
        fake_qdrant_client.scroll.return_value = ([self._fake_point("a", versiyon=5)], None)
        token = _service_token(private_pem)

        resp = client.request(
            "DELETE", "/api/v1/documents/rapor.pdf",
            headers={"Authorization": f"Bearer {token}"},
            json={"beklenen_versiyon": 1, "degistiren_kullanici_id": 42},   # gerçek sürüm 5
        )
        assert resp.status_code == 409

    def test_not_found_returns_404(self, app_client):
        client, fake_qdrant_client, private_pem = app_client
        fake_qdrant_client.collection_exists.return_value = False
        token = _service_token(private_pem)

        resp = client.request(
            "DELETE", "/api/v1/documents/yok.pdf",
            headers={"Authorization": f"Bearer {token}"},
            json={"beklenen_versiyon": 1, "degistiren_kullanici_id": 42},
        )
        assert resp.status_code == 404


class TestDocumentLifecycleRequiresServiceToken:
    def test_user_token_cannot_create_document(self, app_client):
        client, _, private_pem = app_client
        user_token = pyjwt.encode({
            "iss": "bilimp-teracity", "aud": "tubitak1505-query",
            "sub": "613", "iat": int(time.time()), "exp": int(time.time()) + 3600,
            "user_context": {"musteri_id": 501, "sirket_ids": [14], "kullanici_id": 613, "grup_ids": []},
        }, private_pem, algorithm="RS256")

        resp = client.post(
            "/api/v1/documents",
            headers={"Authorization": f"Bearer {user_token}"},
            json={
                "dokuman_id": "rapor.pdf",
                "dosya_icerigi_base64": base64.b64encode(b"icerik").decode(),
                "audience_policy": {"rules": []},
                "yukleyen_kullanici_id": 42,
            },
        )
        assert resp.status_code == 403

    def test_service_token_without_musteri_id_claim_rejected(self, app_client):
        client, _, private_pem = app_client
        token_no_musteri = pyjwt.encode({
            "iss": "bilimp-teracity", "aud": "tubitak1505-audience-admin",
            "sub": "bilimp-backend",
            "iat": int(time.time()), "exp": int(time.time()) + 3600,
        }, private_pem, algorithm="RS256")

        resp = client.delete(
            "/api/v1/documents/rapor.pdf",
            headers={"Authorization": f"Bearer {token_no_musteri}"},
        )
        assert resp.status_code == 401