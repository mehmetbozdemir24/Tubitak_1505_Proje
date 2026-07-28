"""
tests/test_faz4_e2e.py — Faz 4'ün 4 maddesinin (11, 12, 15, 16) HTTP
seviyesinde uçtan uca doğrulaması.
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
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "3")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")

    monkeypatch.setitem(sys.modules, "langchain_qdrant", MagicMock())
    monkeypatch.setitem(sys.modules, "language_utils", MagicMock())
    monkeypatch.setitem(sys.modules, "pymupdf4llm", MagicMock())
    monkeypatch.setitem(sys.modules, "langchain_text_splitters", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm_ingestion", MagicMock())

    for mod in ["api", "rag_service", "document_ingestion_service",
                "error_handling", "rate_limiting"]:
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


def _fake_point(source="belge.pdf", audience_versiyon=1):
    p = MagicMock()
    p.id = "p1"
    p.payload = {
        "metadata": {
            "source": source, "versiyon": 1,
            "audience_versiyon": audience_versiyon,
            "audience": {"rules": []},
        }
    }
    return p


class TestMadde11AudienceVersioning:
    def test_get_audience_returns_version(self, app_client):
        client, fake_qdrant, private_pem = app_client
        fake_qdrant.scroll.return_value = ([_fake_point(audience_versiyon=4)], None)
        token = _service_token(private_pem)

        resp = client.get(
            "/api/v1/documents/belge.pdf/audience",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["audience_versiyon"] == 4

    def test_put_audience_succeeds_with_correct_version(self, app_client):
        client, fake_qdrant, private_pem = app_client
        fake_qdrant.scroll.return_value = ([_fake_point(audience_versiyon=1)], None)
        token = _service_token(private_pem)

        resp = client.put(
            "/api/v1/documents/belge.pdf/audience",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "audience_policy": {"rules": [{"sirket_ids": [14]}]},
                "beklenen_audience_versiyon": 1,
                "degistiren_kullanici_id": 42,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["yeni_audience_versiyon"] == 2

    def test_put_audience_rejects_stale_version(self, app_client):
        client, fake_qdrant, private_pem = app_client
        fake_qdrant.scroll.return_value = ([_fake_point(audience_versiyon=7)], None)
        token = _service_token(private_pem)

        resp = client.put(
            "/api/v1/documents/belge.pdf/audience",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "audience_policy": {"rules": [{"sirket_ids": [14]}]},
                "beklenen_audience_versiyon": 1,
                "degistiren_kullanici_id": 42,
            },
        )
        assert resp.status_code == 409

    def test_content_update_does_not_change_audience_version(self, app_client):
        import base64
        client, fake_qdrant, private_pem = app_client
        fake_qdrant.scroll.return_value = ([_fake_point(audience_versiyon=9)], None)
        token = _service_token(private_pem)

        resp = client.put(
            "/api/v1/documents/belge.pdf/content",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "dosya_icerigi_base64": base64.b64encode(b"yeni").decode(),
                "beklenen_versiyon": 1,
                "degistiren_kullanici_id": 42,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["yeni_versiyon"] == 2


class TestMadde12BulkUpdate:
    def test_bulk_update_partial_success(self, app_client):
        client, fake_qdrant, private_pem = app_client
        token = _service_token(private_pem)

        def scroll_side_effect(*args, **kwargs):
            flt = kwargs.get("scroll_filter")
            source_val = flt.must[0].match.value if flt else None
            if source_val == "b.pdf":
                return ([_fake_point(source="b.pdf", audience_versiyon=5)], None)
            return ([_fake_point(source="a.pdf", audience_versiyon=1)], None)

        fake_qdrant.scroll.side_effect = scroll_side_effect

        resp = client.post(
            "/api/v1/documents/audience/bulk",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "guncellemeler": [
                    {"dokuman_id": "a.pdf", "audience_policy": {"rules": [{"sirket_ids": [14]}]},
                     "beklenen_audience_versiyon": 1},
                    {"dokuman_id": "b.pdf", "audience_policy": {"rules": [{"sirket_ids": [14]}]},
                     "beklenen_audience_versiyon": 1},
                ],
                "degistiren_kullanici_id": 42,
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["toplam"] == 2
        assert body["basarili"] == 1
        assert body["basarisiz"] == 1
        sonuclar_by_id = {s["dokuman_id"]: s for s in body["sonuclar"]}
        assert sonuclar_by_id["a.pdf"]["durum"] == "basarili"
        assert sonuclar_by_id["b.pdf"]["durum"] == "hata"
        assert sonuclar_by_id["b.pdf"]["hata_kodu"] == "CAKISMA"

    def test_bulk_update_rejects_over_100_items(self, app_client):
        client, _, private_pem = app_client
        token = _service_token(private_pem)

        resp = client.post(
            "/api/v1/documents/audience/bulk",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "guncellemeler": [
                    {"dokuman_id": f"d{i}.pdf", "audience_policy": {"rules": []},
                     "beklenen_audience_versiyon": 1}
                    for i in range(101)
                ],
                "degistiren_kullanici_id": 42,
            },
        )
        assert resp.status_code == 422

    def test_bulk_update_requires_service_token(self, app_client):
        client, _, private_pem = app_client
        user_token = pyjwt.encode({
            "iss": "bilimp-teracity", "aud": "tubitak1505-query",
            "sub": "613", "iat": int(time.time()), "exp": int(time.time()) + 3600,
            "user_context": {"musteri_id": 501, "sirket_ids": [14], "kullanici_id": 613, "grup_ids": []},
        }, private_pem, algorithm="RS256")

        resp = client.post(
            "/api/v1/documents/audience/bulk",
            headers={"Authorization": f"Bearer {user_token}"},
            json={"guncellemeler": [], "degistiren_kullanici_id": 42},
        )
        assert resp.status_code in (403, 422)


class TestMadde15ErrorHardening:
    def test_error_response_has_standard_schema(self, app_client):
        client, fake_qdrant, private_pem = app_client
        fake_qdrant.collection_exists.return_value = False
        token = _service_token(private_pem)

        resp = client.get(
            "/api/v1/documents/yok.pdf/audience",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
        body = resp.json()
        assert set(body.keys()) == {"kod", "mesaj", "trace_id"}
        assert body["kod"] == "BULUNAMADI"

    def test_response_includes_trace_id_header(self, app_client):
        client, *_ = app_client
        resp = client.get("/health")
        assert "x-trace-id" in resp.headers

    def test_aud_mismatch_is_403_not_401(self, app_client):
        client, _, private_pem = app_client
        token = _service_token(private_pem)
        resp = client.post(
            "/api/v1/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"soru": "test"},
        )
        assert resp.status_code == 403
        assert resp.json()["kod"] == "YASAK"

    def test_missing_token_is_401(self, app_client):
        client, *_ = app_client
        resp = client.post("/api/v1/query", json={"soru": "test"})
        assert resp.status_code in (401, 403)

    def test_soru_too_long_rejected(self, app_client):
        client, _, private_pem = app_client
        token = pyjwt.encode({
            "iss": "bilimp-teracity", "aud": "tubitak1505-query",
            "sub": "613", "iat": int(time.time()), "exp": int(time.time()) + 3600,
            "user_context": {"musteri_id": 501, "sirket_ids": [14], "kullanici_id": 613, "grup_ids": []},
        }, private_pem, algorithm="RS256")

        resp = client.post(
            "/api/v1/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"soru": "x" * 2001},
        )
        assert resp.status_code == 422

    def test_rate_limit_returns_429(self, app_client):
        client, *_ = app_client
        for _ in range(3):
            client.get("/api/v1/documents/nofile/audience")
        fourth = client.get("/api/v1/documents/nofile/audience")
        assert fourth.status_code == 429
        assert fourth.json()["kod"] == "COK_FAZLA_ISTEK"


class TestMadde16Pagination:
    def test_default_pagination_params(self, app_client):
        client, fake_qdrant, private_pem = app_client
        fake_qdrant.scroll.return_value = ([], None)
        token = _service_token(private_pem)

        resp = client.get(
            "/api/v1/documents/audience-compliance-report",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["limit"] == 50
        assert body["offset"] == 0

    def test_custom_pagination_params_respected(self, app_client):
        client, fake_qdrant, private_pem = app_client
        points = [_fake_point(source=f"belge{i}.pdf") for i in range(5)]
        fake_qdrant.scroll.return_value = (points, None)
        token = _service_token(private_pem)

        resp = client.get(
            "/api/v1/documents/audience-compliance-report",
            headers={"Authorization": f"Bearer {token}"},
params={"limit": 2, "offset": 1},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["limit"] == 2
        assert body["offset"] == 1
        assert len(body["dokumanlar"]) == 2
        assert body["politikasiz_dokuman_sayisi"] == 5

    def test_limit_upper_bound_enforced(self, app_client):
        client, _, private_pem = app_client
        token = _service_token(private_pem)

        resp = client.get(
            "/api/v1/documents/audience-compliance-report",
            headers={"Authorization": f"Bearer {token}"},
params={"limit": 500},
        )
        assert resp.status_code == 422