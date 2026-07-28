"""
tests/test_rag_service_tenancy.py — retrieve_authorized_docs'un MÜŞTERİ
(musteri_id) bazlı tenant çözümlemesini doğru yaptığını ve henüz provizyon
edilmemiş bir tenant için hata yerine boş sonuç döndürdüğünü doğrular.

langchain_qdrant ağır bir bağımlılık olduğundan (ve bu test ortamında
kurulu olmayabileceğinden) sahte bir modülle değiştirilir; burada test
edilen şey embedding/LLM davranışı değil, TENANT ÇÖZÜMLEME mantığıdır.
"""
import sys
from unittest.mock import MagicMock
import pytest


@pytest.fixture(autouse=True)
def fake_heavy_deps(monkeypatch):
    fake_qdrant_module = MagicMock()
    fake_store_instance = MagicMock()
    fake_qdrant_module.QdrantVectorStore.return_value = fake_store_instance
    fake_qdrant_module.RetrievalMode.HYBRID = "hybrid"

    monkeypatch.setitem(sys.modules, "langchain_qdrant", fake_qdrant_module)
    monkeypatch.setitem(sys.modules, "language_utils", MagicMock())
    monkeypatch.setitem(sys.modules, "langchain_core.messages", MagicMock())

    sys.modules.pop("rag_service", None)

    yield fake_store_instance


class TestTenantResolutionInRetrieval:
    def test_query_goes_to_correct_customer_collection(self, fake_heavy_deps):
        import rag_service
        from abac import UserContext
        from tenancy import ConventionTenantRegistry

        fake_store = fake_heavy_deps
        fake_store.similarity_search_with_score.return_value = []

        client = MagicMock()
        client.collection_exists.return_value = True

        user = UserContext(musteri_id=501, sirket_ids=[14], bina_id=16)
        rag_service.retrieve_authorized_docs(
            client, ConventionTenantRegistry(), dense_embeddings=MagicMock(),
            sparse_embeddings=MagicMock(), user=user, question="test",
            top_k=5, threshold=0.3,
        )

        import langchain_qdrant
        _, kwargs = langchain_qdrant.QdrantVectorStore.call_args
        assert kwargs["collection_name"] == "tubitak1505_musteri_501"

    def test_different_customers_never_share_a_collection_call(self, fake_heavy_deps):
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
            UserContext(musteri_id=501, sirket_ids=[14]), "soru1", 5, 0.3,
        )
        rag_service.retrieve_authorized_docs(
            client, registry, MagicMock(), MagicMock(),
            UserContext(musteri_id=777, sirket_ids=[14]), "soru2", 5, 0.3,
        )

        import langchain_qdrant
        collections_used = [
            c.kwargs["collection_name"] for c in langchain_qdrant.QdrantVectorStore.call_args_list
        ]
        assert collections_used == ["tubitak1505_musteri_501", "tubitak1505_musteri_777"]

    def test_unprovisioned_customer_returns_empty_not_error(self, fake_heavy_deps):
        import rag_service
        from abac import UserContext
        from tenancy import ConventionTenantRegistry

        client = MagicMock()
        client.collection_exists.return_value = False

        docs = rag_service.retrieve_authorized_docs(
            client, ConventionTenantRegistry(), MagicMock(), MagicMock(),
            UserContext(musteri_id=999), "soru", 5, 0.3,
        )
        assert docs == []

    def test_sirket_ids_now_fully_participates_in_abac_filter(self, fake_heavy_deps):
        import rag_service
        from abac import UserContext
        from tenancy import ConventionTenantRegistry

        fake_store = fake_heavy_deps
        fake_store.similarity_search_with_score.return_value = []
        client = MagicMock()
        client.collection_exists.return_value = True

        rag_service.retrieve_authorized_docs(
            client, ConventionTenantRegistry(), MagicMock(), MagicMock(),
            UserContext(musteri_id=501, sirket_ids=[14], bina_id=16), "soru", 5, 0.3,
        )

        _, call_kwargs = fake_store.similarity_search_with_score.call_args
        assert "sirket_ids" in str(call_kwargs["filter"])

    # Not: "musteri_id eksik -> 400" senaryosu artik BU KATMANDA test
    # edilemez - UserContext.musteri_id Pydantic'te ZORUNLU bir alandir,
    # yani musteri_id=None ile gecerli bir UserContext hic kurulamaz; bu
    # durum artik auth.py'de (JWT ayristirma aninda, 401 ile) yakalanir.
    # resolve_tenant_collection(None) davranisi tests/test_tenancy.py'de
    # dogrudan test edilmektedir.