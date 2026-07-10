"""
tests/test_migration.py — migrate_to_tenant_collections.py için testler.

Gerçek 12 dokümanlık örnek veri setine (daha önce docker exec ile Qdrant'tan
çektiğimiz) benzer bir sahte veri kümesiyle çalışır. Özellikle şu iki kritik
senaryoyu doğrular:
  1. Tek şirkete özel doküman → tek tenant koleksiyonuna gider.
  2. BİRDEN FAZLA şirkete açık doküman (örn. Eğitim Hafta1.pdf, sirket_ids
     [14, 18, 23]) → İLGİLİ TÜM tenant koleksiyonlarına kopyalanır.
  3. sirket_ids hiç belirtilmeyen doküman → --default-sirket-id'ye düşer VE
     raporlanır.
"""
from unittest.mock import MagicMock
from types import SimpleNamespace

from qdrant_client.http.models import SparseVector

from migrate_to_tenant_collections import _target_sirket_ids, _collect_documents, migrate


def _fake_point(point_id, source, rules):
    return SimpleNamespace(
        id=point_id,
        vector={"content": [0.1, 0.2], "sparse": SparseVector(indices=[1, 2], values=[0.5, 0.5])},
        payload={"metadata": {"source": source, "audience": {"rules": rules}}},
    )


class TestTargetSirketIds:
    def test_single_company_document(self):
        points = [_fake_point(1, "Haftalik_Yemek_Listesi.docx", [{"sirket_ids": [14]}])]
        sirket_ids, used_default = _target_sirket_ids(points, default_sirket_id=99)
        assert sirket_ids == {14}
        assert used_default is False

    def test_multi_company_document_targets_all(self):
        """Eğitim Hafta1.pdf senaryosu: 3 farklı kuralda 3 farklı sirket_id."""
        points = [
            _fake_point(1, "Egitim_Hafta1.pdf", [{"sirket_ids": [14]}]),
            _fake_point(2, "Egitim_Hafta1.pdf", [{"sirket_ids": [18]}]),
            _fake_point(3, "Egitim_Hafta1.pdf", [{"sirket_ids": [23]}]),
        ]
        sirket_ids, used_default = _target_sirket_ids(points, default_sirket_id=99)
        assert sirket_ids == {14, 18, 23}
        assert used_default is False

    def test_single_rule_with_multiple_sirket_ids(self):
        """Tek kuralda sirket_ids=[14,18,23] olan durum da aynı sonucu vermeli."""
        points = [_fake_point(1, "Egitim_Hafta1.pdf", [{"sirket_ids": [14, 18, 23]}])]
        sirket_ids, used_default = _target_sirket_ids(points, default_sirket_id=99)
        assert sirket_ids == {14, 18, 23}

    def test_no_sirket_ids_falls_back_to_default(self):
        """Gerçek test verisinin çoğunluğu (örn. mudurluk_ids ile kısıtlı belgeler)."""
        points = [_fake_point(1, "Butce_Ve_Maliyet_Raporu.docx",
                               [{"mudurluk_ids": [13], "grup_ids": [101]}])]
        sirket_ids, used_default = _target_sirket_ids(points, default_sirket_id=14)
        assert sirket_ids == {14}
        assert used_default is True

    def test_empty_rules_falls_back_to_default(self):
        points = [_fake_point(1, "Bos_Politika.pdf", [])]
        sirket_ids, used_default = _target_sirket_ids(points, default_sirket_id=14)
        assert sirket_ids == {14}
        assert used_default is True


class TestCollectDocuments:
    def test_groups_points_by_source_filename(self):
        client = MagicMock()
        client.scroll.side_effect = [
            (
                [
                    _fake_point(1, "a.pdf", [{"sirket_ids": [14]}]),
                    _fake_point(2, "a.pdf", [{"sirket_ids": [14]}]),
                    _fake_point(3, "b.pdf", [{"sirket_ids": [18]}]),
                ],
                None,  # next_page yok — tek sayfa
            ),
        ]
        docs = _collect_documents(client, "kaynak")
        assert set(docs.keys()) == {"a.pdf", "b.pdf"}
        assert len(docs["a.pdf"]) == 2
        assert len(docs["b.pdf"]) == 1


class TestMigratePlanning:
    def _make_client_with_dataset(self):
        """12 dokümanlık gerçekçi bir örnek küme (çoğu sirket_ids belirtmiyor)."""
        points = [
            _fake_point(1, "183186.cleaned.pdf", [{"sirket_ids": [14]}]),
            _fake_point(2, "Haftalik_Yemek_Listesi.docx", [{"sirket_ids": [14]}]),
            _fake_point(3, "Egitim_Hafta1.pdf", [{"sirket_ids": [14, 18, 23]}]),
            _fake_point(4, "61920.pdf", [{"kullanici_ids": [613]}]),          # sirket_ids YOK
            _fake_point(5, "Butce_Ve_Maliyet_Raporu.docx",
                        [{"mudurluk_ids": [13], "grup_ids": [101]}]),         # sirket_ids YOK
        ]
        client = MagicMock()
        client.scroll.side_effect = [(points, None)]
        return client

    def test_dry_run_does_not_write(self):
        client = self._make_client_with_dataset()
        migrate(client, source="Tubitak_Dokumanlar_Hybrid", default_sirket_id=14, dry_run=True)
        client.upsert.assert_not_called()
        client.create_collection.assert_not_called()

    def test_real_run_writes_to_correct_tenant_collections(self):
        client = self._make_client_with_dataset()
        client.collection_exists.return_value = True  # tüm hedef koleksiyonlar zaten var say

        migrate(client, source="Tubitak_Dokumanlar_Hybrid", default_sirket_id=14, dry_run=False)

        written_collections = {c.kwargs["collection_name"] for c in client.upsert.call_args_list}
        assert written_collections == {
            "tubitak1505_sirket_14",
            "tubitak1505_sirket_18",
            "tubitak1505_sirket_23",
        }

    def test_multi_company_document_written_to_all_three_collections(self):
        client = self._make_client_with_dataset()
        client.collection_exists.return_value = True

        migrate(client, source="Tubitak_Dokumanlar_Hybrid", default_sirket_id=14, dry_run=False)

        for call in client.upsert.call_args_list:
            collection = call.kwargs["collection_name"]
            written_ids = {p.id for p in call.kwargs["points"]}
            if collection == "tubitak1505_sirket_18":
                # Yalnızca Egitim_Hafta1.pdf'in noktası (id=3) buraya gitmeli.
                assert written_ids == {3}
            if collection == "tubitak1505_sirket_23":
                assert written_ids == {3}

    def test_documents_without_sirket_ids_go_to_default_tenant(self):
        client = self._make_client_with_dataset()
        client.collection_exists.return_value = True

        migrate(client, source="Tubitak_Dokumanlar_Hybrid", default_sirket_id=14, dry_run=False)

        default_call = next(
            c for c in client.upsert.call_args_list
            if c.kwargs["collection_name"] == "tubitak1505_sirket_14"
        )
        written_ids = {p.id for p in default_call.kwargs["points"]}
        # 183186 (1), Yemek (2), Egitim_Hafta1 (3), 61920 (4, varsayılan), Butce (5, varsayılan)
        assert written_ids == {1, 2, 3, 4, 5}