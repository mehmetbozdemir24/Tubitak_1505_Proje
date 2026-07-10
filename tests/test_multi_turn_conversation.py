"""
tests/test_multi_turn_conversation.py — Çok turlu konuşma desteğinin
retrieval-sorgu zenginleştirme mantığını ve run_rag_query'ye history
akışının doğru bağlandığını test eder.
"""
import sys
from unittest.mock import MagicMock
import pytest


@pytest.fixture(autouse=True)
def fake_heavy_deps(monkeypatch):
    fake_qdrant_module = MagicMock()
    fake_store = MagicMock()
    fake_store.similarity_search_with_score.return_value = []
    fake_qdrant_module.QdrantVectorStore.return_value = fake_store
    fake_qdrant_module.RetrievalMode.HYBRID = "hybrid"

    monkeypatch.setitem(sys.modules, "langchain_qdrant", fake_qdrant_module)
    monkeypatch.setitem(sys.modules, "language_utils", MagicMock())
    sys.modules.pop("rag_service", None)
    yield fake_store


class TestBuildRetrievalQuery:
    def test_no_history_returns_question_unchanged(self, fake_heavy_deps):
        import rag_service
        result = rag_service._build_retrieval_query("Bu hafta yemekte ne var?", None)
        assert result == "Bu hafta yemekte ne var?"

    def test_empty_history_returns_question_unchanged(self, fake_heavy_deps):
        import rag_service
        result = rag_service._build_retrieval_query("Bu hafta yemekte ne var?", [])
        assert result == "Bu hafta yemekte ne var?"

    def test_follow_up_question_enriched_with_prior_user_turn(self, fake_heavy_deps):
        import rag_service
        from langchain_core.messages import HumanMessage, AIMessage

        history = [
            HumanMessage(content="Bu hafta yemek listesinde ne var?"),
            AIMessage(content="Pazartesi: Kuru fasulye, Salı: Izgara tavuk..."),
        ]
        result = rag_service._build_retrieval_query("Peki salı günü?", history)

        assert "Bu hafta yemek listesinde ne var?" in result
        assert "Peki salı günü?" in result

    def test_only_last_two_user_turns_used(self, fake_heavy_deps):
        import rag_service
        from langchain_core.messages import HumanMessage, AIMessage

        history = [
            HumanMessage(content="ESKİ_TUR_1"),
            AIMessage(content="cevap1"),
            HumanMessage(content="ESKİ_TUR_2"),
            AIMessage(content="cevap2"),
            HumanMessage(content="SON_TUR"),
            AIMessage(content="cevap3"),
        ]
        result = rag_service._build_retrieval_query("güncel soru", history)

        assert "ESKİ_TUR_1" not in result
        assert "ESKİ_TUR_2" in result
        assert "SON_TUR" in result

    def test_history_with_only_ai_messages_falls_back_to_question(self, fake_heavy_deps):
        import rag_service
        from langchain_core.messages import AIMessage

        history = [AIMessage(content="sistem karşılama mesajı")]
        result = rag_service._build_retrieval_query("soru", history)
        assert result == "soru"

    def test_original_question_not_mutated_for_answer_generation(self, fake_heavy_deps):
        import rag_service
        from unittest.mock import patch
        from langchain_core.messages import HumanMessage

        history = [HumanMessage(content="Bu hafta yemek listesi nedir?")]

        with patch("rag_service.answer_with_context") as mock_answer, \
             patch("rag_service.retrieve_authorized_docs", return_value=[MagicMock()]):
            mock_answer.return_value = MagicMock(content="yanıt")
            rag_service.run_rag_query(
                llm=MagicMock(), client=MagicMock(),
                tenant_registry=MagicMock(), dense_embeddings=MagicMock(),
                sparse_embeddings=MagicMock(),
                user=MagicMock(sirket_id=14), question="Peki salı günü?",
                top_k=5, threshold=0.3, history=history,
            )

        call_args = mock_answer.call_args
        assert call_args.args[2] == "Peki salı günü?"


class TestRunRagQueryHistoryWiring:
    def test_history_none_behaves_as_single_turn(self, fake_heavy_deps):
        import rag_service
        from unittest.mock import patch

        with patch("rag_service.answer_with_context") as mock_answer, \
             patch("rag_service.retrieve_authorized_docs", return_value=[MagicMock()]):
            mock_answer.return_value = MagicMock(content="yanıt")
            rag_service.run_rag_query(
                llm=MagicMock(), client=MagicMock(),
                tenant_registry=MagicMock(), dense_embeddings=MagicMock(),
                sparse_embeddings=MagicMock(),
                user=MagicMock(sirket_id=14), question="soru",
                top_k=5, threshold=0.3,
            )
        _, kwargs = mock_answer.call_args
        assert kwargs.get("history") is None

    def test_history_passed_through_to_answer_generation(self, fake_heavy_deps):
        import rag_service
        from unittest.mock import patch
        from langchain_core.messages import HumanMessage

        history = [HumanMessage(content="önceki soru")]
        with patch("rag_service.answer_with_context") as mock_answer, \
             patch("rag_service.retrieve_authorized_docs", return_value=[MagicMock()]):
            mock_answer.return_value = MagicMock(content="yanıt")
            rag_service.run_rag_query(
                llm=MagicMock(), client=MagicMock(),
                tenant_registry=MagicMock(), dense_embeddings=MagicMock(),
                sparse_embeddings=MagicMock(),
                user=MagicMock(sirket_id=14), question="takip sorusu",
                top_k=5, threshold=0.3, history=history,
            )
        _, kwargs = mock_answer.call_args
        assert kwargs.get("history") == history