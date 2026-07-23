"""
tests/test_query_enrichment_e2e.py — Faz 3'ün HTTP seviyesinde doğrulaması:
çok turlu konuşma isteği, kaynaklar[].versiyon, ve uyum raporunun
yapılandırılmış dokumanlar[] şeması.
"""
import sys
import time
from unittest.mock import MagicMock

import pytest
import jwt as pyjwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from langchain_core.documents import Document


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

    fake_language_utils = MagicMock()
    fake_language_utils.choose_answer_language.return_value = ("tr", "tr", "tr", "default")
    fake_language_utils.build_language_policy_prompt.return_value = "dil politikası"
    monkeypatch.setitem(sys.modules, "language_utils", fake_language_utils)

    monkeypatch.setitem(sys.modules, "pymupdf4llm", MagicMock())
    monkeypatch.setitem(sys.modules, "langchain_text_splitters", MagicMock())
    monkeypatch.setitem(sys.modules, "chunker", MagicMock())

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

    return TestClient(api.app), fake_client, fake_store, private_pem


def _user_token(private_pem, sirket_id=14, kullanici_id=613):
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


class TestMultiTurnConversationEndpoint:
    def test_query_without_history_still_works(self, app_client):
        client, fake_qdrant_client, fake_store, private_pem = app_client
        fake_store.similarity_search_with_score.return_value = [
            (Document(page_content="içerik", metadata={"source": "menu.pdf", "versiyon": 1, "page": 1}), 0.8)
        ]
        token = _user_token(private_pem)

        resp = client.post(
            "/api/v1/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"soru": "Bu hafta yemekte ne var?"},
        )
        assert resp.status_code == 200
        assert resp.json()["durum"] == "basarili"

    def test_query_with_history_enriches_retrieval(self, app_client):
        client, fake_qdrant_client, fake_store, private_pem = app_client
        fake_store.similarity_search_with_score.return_value = []
        token = _user_token(private_pem)

        client.post(
            "/api/v1/query",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "soru": "Peki salı günü?",
                "gecmis_mesajlar": [
                    {"rol": "kullanici", "icerik": "Bu hafta yemek listesinde ne var?"},
                    {"rol": "asistan", "icerik": "Pazartesi: Kuru fasulye..."},
                ],
            },
        )

        call_args, _ = fake_store.similarity_search_with_score.call_args
        retrieval_query = call_args[0]
        assert "Bu hafta yemek listesinde ne var?" in retrieval_query
        assert "Peki salı günü?" in retrieval_query

    def test_too_many_history_messages_rejected(self, app_client):
        client, _, _, private_pem = app_client
        token = _user_token(private_pem)

        resp = client.post(
            "/api/v1/query",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "soru": "soru",
                "gecmis_mesajlar": [{"rol": "kullanici", "icerik": "x"}] * 21,
            },
        )
        assert resp.status_code == 422


class TestQuerySourceVersionField:
    def test_kaynaklar_includes_versiyon(self, app_client):
        client, fake_qdrant_client, fake_store, private_pem = app_client
        fake_store.similarity_search_with_score.return_value = [
            (Document(page_content="içerik", metadata={"source": "rapor.pdf", "versiyon": 3, "page": 2}), 0.75)
        ]
        token = _user_token(private_pem)

        resp = client.post(
            "/api/v1/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"soru": "test"},
        )

        assert resp.status_code == 200
        kaynak = resp.json()["kaynaklar"][0]
        assert kaynak["dokuman_id"] == "rapor.pdf"
        assert kaynak["versiyon"] == 3


class TestComplianceReportStructuredItems:
    def test_dokumanlar_returns_structured_objects(self, app_client):
        client, fake_qdrant_client, _, private_pem = app_client
        fake_qdrant_client.collection_exists.return_value = True
        fake_point = MagicMock()
        fake_point.payload = {
            "metadata": {"source": "politikasiz.pdf", "versiyon": 2, "audience": {"rules": []}}
        }
        fake_qdrant_client.scroll.return_value = ([fake_point], None)
        token = _service_token(private_pem)

        resp = client.get(
            "/api/v1/documents/audience-compliance-report",
            headers={"Authorization": f"Bearer {token}"},
            params={"sirket_id": 14},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["politikasiz_dokuman_sayisi"] == 1
        assert body["dokumanlar"][0] == {"dokuman_id": "politikasiz.pdf", "versiyon": 2}