"""
tests/test_document_ingestion.py — document_ingestion_service.py için testler.

chunker.py (gerçek PDF/DOCX ayrıştırma) mock'lanır — burada test edilen şey
dosya ayrıştırma değil, yaşam döngüsü mantığıdır (versiyon, optimistic
locking, politika korunumu, tekrar-oluşturma reddi).
"""
from unittest.mock import MagicMock, patch
import sys
import pytest
from langchain_core.documents import Document

# langchain_qdrant ve chunker'ın kendi ağır bağımlılıkları (pymupdf4llm,
# langchain_text_splitters) bu test ortamında kurulu olmayabilir — üretimde
# requirements.txt üzerinden zaten kurulu olacaklar. Burada test edilen şey
# dosya ayrıştırma değil, yaşam döngüsü mantığı olduğundan sahteleriyle
# değiştirmek yeterli ve doğrudur. Modül import edilmeden ÖNCE yapılmalı.
sys.modules.setdefault("langchain_qdrant", MagicMock())
sys.modules.setdefault("pymupdf4llm", MagicMock())
sys.modules.setdefault("langchain_text_splitters", MagicMock())

from document_ingestion_service import (
    create_document,
    update_document_content,
    delete_document,
    DocumentIngestionError,
    MAX_FILE_SIZE_BYTES,
)
from abac import AudiencePolicy, AudienceRule


def _fake_chunks(n=2):
    return [Document(page_content=f"içerik {i}", metadata={}) for i in range(n)]


class TestCreateDocument:
    def test_successful_creation(self):
        client = MagicMock()
        client.collection_exists.return_value = True   # tenant koleksiyonu zaten var
        client.scroll.return_value = ([], None)

        with patch("document_ingestion_service.chunk_text", return_value=_fake_chunks(3)), \
             patch("document_ingestion_service.QdrantVectorStore") as MockStore:
            mock_store = MockStore.return_value
            result = create_document(
                client=client, collection="tubitak1505_musteri_501",
                dokuman_id="rapor.pdf", file_bytes=b"sahte pdf icerigi",
                file_ext=".pdf",
                audience_policy=AudiencePolicy(rules=[AudienceRule(sirket_ids=[14])]),
                dense_embeddings=MagicMock(), sparse_embeddings=MagicMock(),
                reference_collection="Tubitak_Dokumanlar_Hybrid",
            )

        assert result.dokuman_id == "rapor.pdf"
        assert result.versiyon == 1
        assert result.chunk_sayisi == 3
        mock_store.add_documents.assert_called_once()

    def test_provisions_tenant_collection_if_missing(self):
        """
        (Bug fix) Bu müşteri için ilk doküman yükleniyorsa (koleksiyon henüz
        yoksa), referans koleksiyonun şeması kopyalanarak otomatik
        oluşturulmalı — QdrantVectorStore kurulumu 404 ile çökmemeli.
        """
        client = MagicMock()

        def collection_exists_side_effect(name):
            return name != "tubitak1505_musteri_999"   # yalnızca tenant koleksiyonu yok

        client.collection_exists.side_effect = collection_exists_side_effect
        client.scroll.return_value = ([], None)

        fake_ref = MagicMock()
        fake_ref.config.params.vectors = {"content": "FAKE_DENSE"}
        fake_ref.config.params.sparse_vectors = {"sparse": "FAKE_SPARSE"}
        client.get_collection.return_value = fake_ref

        with patch("document_ingestion_service.chunk_text", return_value=_fake_chunks(1)), \
             patch("document_ingestion_service.QdrantVectorStore"):
            create_document(
                client=client, collection="tubitak1505_musteri_999",
                dokuman_id="ilk_belge.pdf", file_bytes=b"icerik", file_ext=".pdf",
                audience_policy=AudiencePolicy(rules=[AudienceRule(sirket_ids=[99])]),
                dense_embeddings=MagicMock(), sparse_embeddings=MagicMock(),
                reference_collection="Tubitak_Dokumanlar_Hybrid",
            )

        client.create_collection.assert_called_once_with(
            collection_name="tubitak1505_musteri_999",
            vectors_config={"content": "FAKE_DENSE"},
            sparse_vectors_config={"sparse": "FAKE_SPARSE"},
        )

    def test_rejects_duplicate_dokuman_id(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        existing_point = MagicMock()
        client.scroll.return_value = ([existing_point], None)

        with pytest.raises(DocumentIngestionError) as exc_info:
            create_document(
                client=client, collection="tubitak1505_musteri_501",
                dokuman_id="rapor.pdf", file_bytes=b"icerik", file_ext=".pdf",
                audience_policy=AudiencePolicy(), dense_embeddings=MagicMock(),
                sparse_embeddings=MagicMock(),
                reference_collection="Tubitak_Dokumanlar_Hybrid",
                allow_empty=True,
            )
        assert exc_info.value.code == "already_exists"

    def test_rejects_oversized_file(self):
        client = MagicMock()
        with pytest.raises(DocumentIngestionError) as exc_info:
            create_document(
                client=client, collection="c", dokuman_id="buyuk.pdf",
                file_bytes=b"x" * (MAX_FILE_SIZE_BYTES + 1), file_ext=".pdf",
                audience_policy=AudiencePolicy(), dense_embeddings=MagicMock(),
                sparse_embeddings=MagicMock(),
                reference_collection="Tubitak_Dokumanlar_Hybrid",
            )
        assert exc_info.value.code == "too_large"

    def test_rejects_when_chunker_produces_nothing(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.scroll.return_value = ([], None)

        with patch("document_ingestion_service.chunk_text", return_value=[]):
            with pytest.raises(DocumentIngestionError) as exc_info:
                create_document(
                    client=client, collection="c", dokuman_id="bozuk.pdf",
                    file_bytes=b"gecersiz", file_ext=".pdf",
                    audience_policy=AudiencePolicy(), dense_embeddings=MagicMock(),
                    sparse_embeddings=MagicMock(),
                    reference_collection="Tubitak_Dokumanlar_Hybrid",
                    allow_empty=True,
                )
        assert exc_info.value.code == "chunking_failed"

    def test_pptx_uses_pptx_chunker(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.scroll.return_value = ([], None)

        with patch("document_ingestion_service.chunk_pptx", return_value=_fake_chunks(1)) as mock_pptx, \
             patch("document_ingestion_service.chunk_text") as mock_text, \
             patch("document_ingestion_service.QdrantVectorStore"):
            create_document(
                client=client, collection="c", dokuman_id="sunum.pptx",
                file_bytes=b"icerik", file_ext=".pptx",
                audience_policy=AudiencePolicy(), dense_embeddings=MagicMock(),
                sparse_embeddings=MagicMock(),
                reference_collection="Tubitak_Dokumanlar_Hybrid",
                allow_empty=True,
            )
        mock_pptx.assert_called_once()
        mock_text.assert_not_called()


class TestUpdateDocumentContent:
    def _client_with_existing_doc(self, versiyon=1, audience=None):
        client = MagicMock()
        client.collection_exists.return_value = True
        existing_point = MagicMock()
        existing_point.id = "p1"
        existing_point.payload = {
            "metadata": {"versiyon": versiyon, "audience": audience or {"rules": [{"sirket_ids": [14]}]}}
        }
        client.scroll.return_value = ([existing_point], None)
        return client

    def test_successful_update_increments_version(self):
        client = self._client_with_existing_doc(versiyon=1)

        with patch("document_ingestion_service.chunk_text", return_value=_fake_chunks(2)), \
             patch("document_ingestion_service.QdrantVectorStore"):
            result = update_document_content(
                client=client, collection="c", dokuman_id="rapor.pdf",
                file_bytes=b"yeni icerik", file_ext=".pdf", beklenen_versiyon=1,
                dense_embeddings=MagicMock(), sparse_embeddings=MagicMock(),
            )

        assert result.versiyon == 2

    def test_deletes_old_points_after_successful_update(self):
        client = self._client_with_existing_doc(versiyon=1)

        with patch("document_ingestion_service.chunk_text", return_value=_fake_chunks(2)), \
             patch("document_ingestion_service.QdrantVectorStore"):
            update_document_content(
                client=client, collection="c", dokuman_id="rapor.pdf",
                file_bytes=b"yeni icerik", file_ext=".pdf", beklenen_versiyon=1,
                dense_embeddings=MagicMock(), sparse_embeddings=MagicMock(),
            )

        client.delete.assert_called_once()

    def test_preserves_existing_audience_policy(self):
        original_audience = {"rules": [{"mudurluk_ids": [13], "grup_ids": [101]}]}
        client = self._client_with_existing_doc(versiyon=1, audience=original_audience)

        captured_audience = {}

        def fake_chunk_text(path, source, size, overlap, audience):
            captured_audience.update(audience)
            return _fake_chunks(1)

        with patch("document_ingestion_service.chunk_text", side_effect=fake_chunk_text), \
             patch("document_ingestion_service.QdrantVectorStore"):
            update_document_content(
                client=client, collection="c", dokuman_id="rapor.pdf",
                file_bytes=b"yeni icerik", file_ext=".pdf", beklenen_versiyon=1,
                dense_embeddings=MagicMock(), sparse_embeddings=MagicMock(),
            )

        assert captured_audience == original_audience

    def test_rejects_when_document_not_found(self):
        client = MagicMock()
        client.collection_exists.return_value = False

        with pytest.raises(DocumentIngestionError) as exc_info:
            update_document_content(
                client=client, collection="c", dokuman_id="yok.pdf",
                file_bytes=b"icerik", file_ext=".pdf", beklenen_versiyon=1,
                dense_embeddings=MagicMock(), sparse_embeddings=MagicMock(),
            )
        assert exc_info.value.code == "not_found"

    def test_rejects_on_version_mismatch(self):
        client = self._client_with_existing_doc(versiyon=3)

        with pytest.raises(DocumentIngestionError) as exc_info:
            update_document_content(
                client=client, collection="c", dokuman_id="rapor.pdf",
                file_bytes=b"icerik", file_ext=".pdf", beklenen_versiyon=1,
                dense_embeddings=MagicMock(), sparse_embeddings=MagicMock(),
            )
        assert exc_info.value.code == "version_conflict"

    def test_rejects_oversized_file(self):
        client = self._client_with_existing_doc()
        with pytest.raises(DocumentIngestionError) as exc_info:
            update_document_content(
                client=client, collection="c", dokuman_id="rapor.pdf",
                file_bytes=b"x" * (MAX_FILE_SIZE_BYTES + 1), file_ext=".pdf",
                beklenen_versiyon=1, dense_embeddings=MagicMock(), sparse_embeddings=MagicMock(),
            )
        assert exc_info.value.code == "too_large"


class TestDeleteDocument:
    def _fake_point(self, point_id, versiyon=1):
        p = MagicMock()
        p.id = point_id
        p.payload = {"metadata": {"versiyon": versiyon, "source": "rapor.pdf"}}
        return p

    def test_successful_deletion(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        p1, p2 = self._fake_point("a", versiyon=1), self._fake_point("b", versiyon=1)
        client.scroll.return_value = ([p1, p2], None)

        count = delete_document(
            client, "c", "rapor.pdf",
            beklenen_versiyon=1, degistiren_kullanici_id=42,
        )

        assert count == 2
        client.delete.assert_called_once()

    def test_rejects_when_not_found(self):
        client = MagicMock()
        client.collection_exists.return_value = False

        with pytest.raises(DocumentIngestionError) as exc_info:
            delete_document(
                client, "c", "yok.pdf",
                beklenen_versiyon=1, degistiren_kullanici_id=42,
            )
        assert exc_info.value.code == "not_found"

    def test_rejects_on_version_mismatch(self):
        """(madde 5) DELETE artık diğer yazma uçlarıyla aynı optimistic
        locking korumasına tabi."""
        client = MagicMock()
        client.collection_exists.return_value = True
        client.scroll.return_value = ([self._fake_point("a", versiyon=5)], None)

        with pytest.raises(DocumentIngestionError) as exc_info:
            delete_document(
                client, "c", "rapor.pdf",
                beklenen_versiyon=1, degistiren_kullanici_id=42,   # gerçek sürüm 5
            )
        assert exc_info.value.code == "version_conflict"