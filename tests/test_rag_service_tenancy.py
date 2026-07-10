"""
tests/test_rag_service_tenancy.py — retrieve_authorized_docs'un tenant
çözümlemesini doğru yaptığını ve henüz provizyon edilmemiş bir tenant için
hata yerine boş sonuç döndürdüğünü doğrular.

langchain_qdrant ağır bir bağımlılık olduğundan (ve bu test ortamında
kurulu olmayabileceğinden) sahte bir modülle değiştirilir; burada test
edilen şey embedding/LLM davranışı değil, TENANT ÇÖZÜMLEME mantığıdır.
"""
import sys
from unittest.mock import MagicMock
import pytest


@pytest.fixture(autouse=True)
def fake_heavy_deps(monkeypatch):
    """langchain_qdrant ve language_utils'i test boyunca sahteleriyle değiştirir."""
    fake_qdrant_module = MagicMock()
    fake_store_instance = MagicMock()
    fake_qdrant_module.QdrantVectorStore.return_value = fake_store_instance
    fake_qdrant_module.RetrievalMode.HYBRID = "hybrid"

    monkeypatch.setitem(sys.modules, "langchain_qdrant", fake_qdrant_module)
    monkeypatch.setitem(sys.modules, "language_utils", MagicMock())
    monkeypatch.setitem(sys.modules, "langchain_core.messages", MagicMock())

    # rag_service modülü daha önce import edilmiş olabilir (başka testlerden) —
    # sahte bağımlılıklarla temiz şekilde yeniden import edilmesini garanti et.
    sys.modules.pop("rag_service", None)

    yield fake_store_instance


class TestTenantResolutionInRetrieval:
    def test_query_goes_to_correct_tenant_collection(self, fake_heavy_deps):
        import rag_service
        from abac import UserContext
        from tenancy import ConventionTenantRegistry

        fake_store = fake_heavy_deps
        fake_store.similarity_search_with_score.return_value = []

        client = MagicMock()
        client.collection_exists.return_value = True

        user = UserContext(sirket_id=14, bina_id=16)
        rag_service.retrieve_authorized_docs(
            client, ConventionTenantRegistry(), dense_embeddings=MagicMock(),
            sparse_embeddings=MagicMock(), user=user, question="test",
            top_k=5, threshold=0.3,
        )

        # QdrantVectorStore hangi koleksiyonla kuruldu?
        import langchain_qdrant
        _, kwargs = langchain_qdrant.QdrantVectorStore.call_args
        assert kwargs["collection_name"] == "tubitak1505_sirket_14"

    def test_different_tenants_never_share_a_collection_call(self, fake_heavy_deps):
        import rag_service
        from abac import UserContext
        from tenancy import ConventionTenantRegistry

        fake_store = fake_heavy_deps
        fake_store.similarity_search_with_score.return_value = []
        client = MagicMock()
        client.collection_exists.return_value = True
        registry = ConventionTenantRegistry()

        rag_service.retrieve_authorized_docs(
            client, registry, MagicMock(), MagicMock(),
            UserContext(sirket_id=14), "soru1", 5, 0.3,
        )
        rag_service.retrieve_authorized_docs(
            client, registry, MagicMock(), MagicMock(),
            UserContext(sirket_id=18), "soru2", 5, 0.3,
        )

        import langchain_qdrant
        collections_used = [
            c.kwargs["collection_name"] for c in langchain_qdrant.QdrantVectorStore.call_args_list
        ]
        assert collections_used == ["tubitak1505_sirket_14", "tubitak1505_sirket_18"]

    def test_missing_sirket_id_raises_400(self, fake_heavy_deps):
        import rag_service
        from abac import UserContext
        from tenancy import ConventionTenantRegistry
        from fastapi import HTTPException

        client = MagicMock()
        user = UserContext(sirket_id=None)  # JWT'de tenant bilgisi yok

        with pytest.raises(HTTPException) as exc_info:
            rag_service.retrieve_authorized_docs(
                client, ConventionTenantRegistry(), MagicMock(), MagicMock(),
                user, "soru", 5, 0.3,
            )
        assert exc_info.value.status_code == 400

    def test_unprovisioned_tenant_returns_empty_not_error(self, fake_heavy_deps):
        """Henüz hiç dokümanı olmayan (koleksiyonu oluşturulmamış) bir tenant
        için hata değil, boş sonuç dönmeli — deny-by-default ile tutarlı."""
        import rag_service
        from abac import UserContext
        from tenancy import ConventionTenantRegistry

        client = MagicMock()
        client.collection_exists.return_value = False  # tenant koleksiyonu yok

        docs = rag_service.retrieve_authorized_docs(
            client, ConventionTenantRegistry(), MagicMock(), MagicMock(),
            UserContext(sirket_id=999), "soru", 5, 0.3,
        )
        assert docs == []

    def test_sirket_ids_excluded_from_abac_filter_in_tenant_mode(self, fake_heavy_deps):
        """sirket_ids artık koleksiyon içinde anlamsız olduğu için ABAC
        filtresine hiç girmemeli."""
        import rag_service
        from abac import UserContext
        from tenancy import ConventionTenantRegistry

        fake_store = fake_heavy_deps
        fake_store.similarity_search_with_score.return_value = []
        client = MagicMock()
        client.collection_exists.return_value = True

        rag_service.retrieve_authorized_docs(
            client, ConventionTenantRegistry(), MagicMock(), MagicMock(),
            UserContext(sirket_id=14, bina_id=16), "soru", 5, 0.3,
        )

        _, call_kwargs = fake_store.similarity_search_with_score.call_args
        assert "sirket_ids" not in str(call_kwargs["filter"])