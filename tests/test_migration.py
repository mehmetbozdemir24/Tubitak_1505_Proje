"""
tests/test_migration.py — migrate_to_tenant_collections.py (v2.0, musteri_id
bazlı) için testler.
"""
from unittest.mock import MagicMock

from migrate_to_tenant_collections import _collect_documents, migrate


def _fake_point(point_id, source, rules=None):
    from types import SimpleNamespace
    from qdrant_client.http.models import SparseVector
    return SimpleNamespace(
        id=point_id,
        vector={"content": [0.1, 0.2], "sparse": SparseVector(indices=[1, 2], values=[0.5, 0.5])},
        payload={"metadata": {"source": source, "audience": {"rules": rules or []}}},
    )


class TestCollectDocuments:
    def test_groups_points_by_source_filename(self):
        client = MagicMock()
        client.scroll.side_effect = [
            (
                [
                    _fake_point(1, "a.pdf", [{"sirket_ids": [14]}]),
                    _fake_point(2, "a.pdf", [{"sirket_ids": [14]}]),
                    _fake_point(3, "b.pdf", [{"kullanici_ids": [613]}]),
                ],
                None,
            ),
        ]
        docs = _collect_documents(client, "kaynak")
        assert set(docs.keys()) == {"a.pdf", "b.pdf"}
        assert len(docs["a.pdf"]) == 2
        assert len(docs["b.pdf"]) == 1


class TestMigratePlanning:
    def _make_client_with_dataset(self):
        points = [
            _fake_point(1, "183186.cleaned.pdf", [{"sirket_ids": [14]}]),
            _fake_point(2, "Haftalik_Yemek_Listesi.docx", [{"sirket_ids": [14, 18, 23]}]),
            _fake_point(3, "61920.pdf", [{"kullanici_ids": [613]}]),
            _fake_point(4, "Butce_Ve_Maliyet_Raporu.docx",
                        [{"mudurluk_ids": [13], "grup_ids": [101]}]),
        ]
        client = MagicMock()
        client.scroll.side_effect = [(points, None)]
        return client

    def test_dry_run_does_not_write(self):
        client = self._make_client_with_dataset()
        migrate(client, source="Tubitak_Dokumanlar_Hybrid", musteri_id=501, dry_run=True)
        client.upsert.assert_not_called()
        client.create_collection.assert_not_called()

    def test_real_run_writes_all_documents_to_single_customer_collection(self):
        client = self._make_client_with_dataset()
        client.collection_exists.return_value = True

        migrate(client, source="Tubitak_Dokumanlar_Hybrid", musteri_id=501, dry_run=False)

        client.upsert.assert_called_once()
        _, kwargs = client.upsert.call_args
        assert kwargs["collection_name"] == "tubitak1505_musteri_501"
        written_ids = {p.id for p in kwargs["points"]}
        assert written_ids == {1, 2, 3, 4}

    def test_sirket_ids_preserved_as_is_in_payload(self):
        client = self._make_client_with_dataset()
        client.collection_exists.return_value = True

        migrate(client, source="Tubitak_Dokumanlar_Hybrid", musteri_id=501, dry_run=False)

        _, kwargs = client.upsert.call_args
        point_2 = next(p for p in kwargs["points"] if p.id == 2)
        rules = point_2.payload["metadata"]["audience"]["rules"]
        assert rules == [{"sirket_ids": [14, 18, 23]}]

    def test_provisions_target_collection_from_reference_schema(self):
        client = self._make_client_with_dataset()
        client.collection_exists.side_effect = lambda name: name != "tubitak1505_musteri_501"

        fake_ref = MagicMock()
        fake_ref.config.params.vectors = {"content": "DENSE"}
        fake_ref.config.params.sparse_vectors = {"sparse": "SPARSE"}
        client.get_collection.return_value = fake_ref

        migrate(client, source="Tubitak_Dokumanlar_Hybrid", musteri_id=501, dry_run=False)

        client.create_collection.assert_called_once_with(
            collection_name="tubitak1505_musteri_501",
            vectors_config={"content": "DENSE"},
            sparse_vectors_config={"sparse": "SPARSE"},
        )