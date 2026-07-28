"""
tests/test_audience_service_tenancy.py — audience_service.py fonksiyonlarının,
henüz provizyon edilmemiş (var olmayan) bir tenant koleksiyonuyla
karşılaştığında Qdrant'ın kendi istisnasını çıplak fırlatmak yerine temiz
bir sonuç (None / boş sayfa / not_found) döndürdüğünü doğrular.

Faz 4 güncellemesi: update_document_audience artık beklenen_audience_versiyon
parametresi ister (madde 11); find_documents_without_audience artık
(sayfa, toplam) tuple'ı döner (madde 16).
"""
from unittest.mock import MagicMock
import pytest

from audience_service import (
    update_document_audience,
    get_document_audience,
    find_documents_without_audience,
    AudienceUpdateError,
)
from abac import AudiencePolicy, AudienceRule


class TestUnprovisionedTenantCollection:
    def test_update_raises_not_found_when_collection_missing(self):
        client = MagicMock()
        client.collection_exists.return_value = False

        with pytest.raises(AudienceUpdateError) as exc_info:
            update_document_audience(
                client=client,
                collection="tubitak1505_musteri_999",
                source="belge.pdf",
                policy=AudiencePolicy(rules=[AudienceRule(bina_ids=[16])]),
                degistiren_kullanici_id=42,
                beklenen_audience_versiyon=1,
            )
        assert exc_info.value.code == "not_found"

    def test_get_returns_none_when_collection_missing(self):
        client = MagicMock()
        client.collection_exists.return_value = False

        result = get_document_audience(client, "tubitak1505_musteri_999", "belge.pdf")
        assert result is None

    def test_compliance_report_returns_empty_when_collection_missing(self):
        client = MagicMock()
        client.collection_exists.return_value = False

        sayfa, toplam = find_documents_without_audience(client, "tubitak1505_musteri_999")
        assert sayfa == []
        assert toplam == 0

    def test_no_raw_qdrant_exception_leaks_for_any_function(self):
        client = MagicMock()
        client.collection_exists.return_value = False

        get_document_audience(client, "yok", "belge.pdf")
        find_documents_without_audience(client, "yok")
        try:
            update_document_audience(
                client, "yok", "belge.pdf",
                AudiencePolicy(rules=[AudienceRule(bina_ids=[16])]),
                degistiren_kullanici_id=1,
                beklenen_audience_versiyon=1,
            )
        except AudienceUpdateError:
            pass

        client.scroll.assert_not_called()


class TestExistingCollectionBehaviorUnchanged:
    """Koleksiyon varsa temel davranış (bulma/yazma) çalışmaya devam etmeli."""

    def _fake_point(self, versiyon=1, audience_versiyon=1):
        p = MagicMock()
        p.id = "p1"
        p.payload = {
            "metadata": {
                "source": "belge.pdf",
                "versiyon": versiyon,
                "audience_versiyon": audience_versiyon,
                "audience": {"rules": []},
            }
        }
        return p

    def test_update_succeeds_with_correct_expected_version(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.scroll.return_value = ([self._fake_point(audience_versiyon=1)], None)

        yeni_versiyon = update_document_audience(
            client=client,
            collection="tubitak1505_musteri_14",
            source="belge.pdf",
            policy=AudiencePolicy(rules=[AudienceRule(bina_ids=[16])]),
            degistiren_kullanici_id=42,
            beklenen_audience_versiyon=1,
        )
        assert yeni_versiyon == 2

    def test_update_rejects_stale_expected_version(self):
        """(Faz 4 / madde 11) Gerçek optimistic locking testi."""
        client = MagicMock()
        client.collection_exists.return_value = True
        client.scroll.return_value = ([self._fake_point(audience_versiyon=5)], None)

        with pytest.raises(AudienceUpdateError) as exc_info:
            update_document_audience(
                client=client,
                collection="tubitak1505_musteri_14",
                source="belge.pdf",
                policy=AudiencePolicy(rules=[AudienceRule(bina_ids=[16])]),
                degistiren_kullanici_id=42,
                beklenen_audience_versiyon=1,   # gerçek sürüm 5, uyuşmuyor
            )
        assert exc_info.value.code == "version_conflict"

    def test_get_returns_policy_and_version(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.scroll.return_value = ([self._fake_point(audience_versiyon=3)], None)

        info = get_document_audience(client, "tubitak1505_musteri_14", "belge.pdf")
        assert info.audience_versiyon == 3
        assert info.policy == {"rules": []}

    def test_get_also_returns_content_version(self):
        """(Teracity ikinci inceleme, madde 1) GET /audience yanıtı içerik
        sürümünü de içermeli — aksi halde 'PUT /content sonrası 409'dan
        kurtarma akışı (Bölüm 10) kodlanamaz."""
        client = MagicMock()
        client.collection_exists.return_value = True
        client.scroll.return_value = ([self._fake_point(versiyon=7, audience_versiyon=3)], None)

        info = get_document_audience(client, "tubitak1505_musteri_14", "belge.pdf")
        assert info.icerik_versiyonu == 7
        assert info.audience_versiyon == 3   # iki sayaç birbirinden bağımsız

    def test_set_payload_called_with_dict_not_scalar(self):
        """
        (Bug fix) Qdrant'ın gerçek set_payload'ı, 'key' verildiğinde 'payload'
        parametresinin HER ZAMAN bir sözlük olmasını şart koşar (Pydantic
        SetPayload modeli bunu zorunlu kılar) — ham bir int (örn. sadece
        audience_versiyon sayısı) geçersizdir ve gerçek istemcide
        ValidationError fırlatır. MagicMock bunu doğrulamadığı için bu hatayı
        yakalamak için payload'ın gerçekten dict olduğunu açıkça test ediyoruz.
        """
        client = MagicMock()
        client.collection_exists.return_value = True
        client.scroll.return_value = ([self._fake_point(audience_versiyon=1)], None)

        update_document_audience(
            client=client,
            collection="tubitak1505_musteri_14",
            source="belge.pdf",
            policy=AudiencePolicy(rules=[AudienceRule(bina_ids=[16])]),
            degistiren_kullanici_id=42,
            beklenen_audience_versiyon=1,
        )

        for call in client.set_payload.call_args_list:
            assert isinstance(call.kwargs["payload"], dict), (
                f"payload dict olmalı, {type(call.kwargs['payload'])} geldi: "
                f"{call.kwargs['payload']!r}"
            )
        all_payloads = [call.kwargs["payload"] for call in client.set_payload.call_args_list]
        assert any(p.get("audience_versiyon") == 2 for p in all_payloads)