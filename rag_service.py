"""
rag_service.py — RAG çekirdek mantığı (UI'dan bağımsız).

Streamlit sayfası ve FastAPI endpoint'i bu servisi ortak kullanır. Böylece
retrieval + ABAC filtresi + yanıt üretimi tek yerde tanımlanır ve iki arayüz
arasında davranış farkı oluşmaz.

Bu modül Streamlit'e HİÇBİR bağımlılık taşımaz.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from abac import UserContext, build_qdrant_abac_filter
from prompts import build_rag_prompt, build_labeled_context
from language_utils import choose_answer_language, build_language_policy_prompt
from tenancy import TenantRegistry, resolve_tenant_collection

# Seviye C mimarisinde tenant sınırı koleksiyon seçimiyle sağlanır; bu alan
# artık koleksiyon içinde anlamsızdır (bkz. tenancy.py modül dokümantasyonu).
_TENANT_ISOLATED_FIELDS = frozenset({"sirket_ids"})


@dataclass
class RagResult:
    durum: str                       # "basarili" | "sonuc_yok" | "hata"
    yanit: str = ""
    kaynaklar: list = field(default_factory=list)   # retrieved Document listesi


def _build_retrieval_query(question: str, history: list | None) -> str:
    """
    Çok turlu konuşmada, bir takip sorusu ("Peki salı günü?") tek başına
    anlamlı bir arama sorgusu OLMAYABİLİR — önceki turun bağlamı olmadan
    embedding araması alakasız sonuçlar döndürebilir.

    Ekstra bir LLM çağrısı (query-rewriting) yapmadan — gecikme/maliyet
    eklemeden — son kullanıcı turlarını güncel soruyla birleştirerek daha
    isabetli bir arama metni üretir. Bu, YALNIZCA retrieval için kullanılır;
    LLM'e gönderilen asıl soru (answer_with_context'teki 'question')
    DEĞİŞTİRİLMEZ — kullanıcı orijinal ifadesiyle yanıtlanır.
    """
    if not history:
        return question

    from langchain_core.messages import HumanMessage
    recent_user_turns = [m.content for m in history if isinstance(m, HumanMessage)][-2:]
    if not recent_user_turns:
        return question
    return " ".join(recent_user_turns + [question])


def retrieve_authorized_docs(
    client: QdrantClient,
    tenant_registry: TenantRegistry,
    dense_embeddings,
    sparse_embeddings,
    user: UserContext,
    question: str,
    top_k: int,
    threshold: float,
) -> list:
    """
    (RAG madde 9 / güvenlik) ABAC pre-filter ile SADECE kullanıcının erişebildiği
    belgeleri getirir. Erişim kontrolü iki katmanlıdır:
      1. Fiziksel: sorgu yalnızca kullanıcının kendi tenant koleksiyonuna gider.
      2. Mantıksal: koleksiyon içinde ABAC filtresi (şube/müdürlük/bina/...) uygulanır.
    LLM'e asla yetkisiz belge ulaşmaz.

    Not: 'question' parametresi burada DOĞRUDAN arama metni olarak kullanılır
    (çok turlu bağlam zenginleştirmesi varsa çağıran taraf — run_rag_query —
    bunu zaten uygulayıp zenginleştirilmiş metni buraya geçirir). Bu
    fonksiyon "history" kavramını bilerek bilmez (SRP): yalnızca "hangi
    metinle arama yapılacağı" ile ilgilenir.
    """
    collection = resolve_tenant_collection(tenant_registry, user.sirket_id)

    if not client.collection_exists(collection):
        # Tenant için henüz hiç doküman yüklenmemiş (koleksiyon provizyon
        # edilmemiş). Bu bir hata değildir — boş sonuç, run_rag_query
        # tarafından deny-by-default olarak ele alınır.
        return []

    store = QdrantVectorStore(
        client=client, collection_name=collection,
        embedding=dense_embeddings, vector_name="content",
        sparse_embedding=sparse_embeddings, sparse_vector_name="sparse",
        retrieval_mode=RetrievalMode.HYBRID,
    )
    qdrant_filter = build_qdrant_abac_filter(user, exclude_fields=_TENANT_ISOLATED_FIELDS)
    results = store.similarity_search_with_score(question, k=top_k, filter=qdrant_filter)

    docs = []
    for doc, score in results:
        if score >= threshold:
            doc.metadata["score"] = score
            docs.append(doc)
    return docs


def answer_with_context(llm, retrieved_docs: list, question: str, history: list | None = None):
    """
    Bağlamı kaynak-etiketli biçimde kurar ve LLM'i çağırır.
    Streaming DEĞİL — API tarafı için tam yanıt döndürür. (UI streaming'i
    ayrı yönetir; bkz. Streamlit sayfası.)
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    context_str = build_labeled_context(retrieved_docs)
    answer_lang, *_ = choose_answer_language(question, context_str)
    rag_prompt = build_rag_prompt(build_language_policy_prompt(answer_lang), context_str)

    msgs = [SystemMessage(content=rag_prompt)]
    if history:
        msgs += history
    msgs.append(HumanMessage(content=question))
    return llm.invoke(msgs)


def run_rag_query(
    llm,
    client: QdrantClient,
    tenant_registry: TenantRegistry,
    dense_embeddings,
    sparse_embeddings,
    user: UserContext,
    question: str,
    top_k: int,
    threshold: float,
    history: list | None = None,
) -> RagResult:
    """
    Uçtan uca (API için): tenant çözümle → getir → yetki guard-rail → yanıtla.

    history: LangChain HumanMessage/AIMessage listesi (önceki turlar), en
    yeniden en eskiye değil, KRONOLOJİK sırayla. None/boş ise tek turluk
    davranış (Faz 3 öncesiyle birebir aynı).

    (10 / error handling) Erişilebilir belge yoksa LLM TETİKLENMEZ; deny-by-default
    sonucu 'sonuc_yok' döner. Getirme (retrieval) VEYA yanıt üretme sırasında
    oluşan HERHANGİ bir hata 'hata' olarak sarmalanır; iç detay (stack trace,
    Qdrant/embedding iç mesajı) dışarı sızdırılmaz — çağıran katman (api.py)
    genel bir mesaja çevirir. Böylece hiçbir istisna FastAPI'ye çıplak ulaşıp
    500 üretmez; loglanabilir, kontrollü bir sonuç her zaman döner.
    """
    import logging
    logger = logging.getLogger("rag_service")

    retrieval_query = _build_retrieval_query(question, history)

    try:
        docs = retrieve_authorized_docs(
            client, tenant_registry, dense_embeddings, sparse_embeddings,
            user, retrieval_query, top_k, threshold,
        )
    except Exception:
        logger.exception("retrieve_authorized_docs başarısız oldu")
        return RagResult(
            durum="hata",
            yanit="Belgeler alınırken bir sorun oluştu. Lütfen daha sonra tekrar deneyin.",
            kaynaklar=[],
        )

    if not docs:
        return RagResult(
            durum="sonuc_yok",
            yanit="Bu konuda erişim yetkiniz dahilinde bir doküman bulunamadı.",
            kaynaklar=[],
        )

    try:
        ai_msg = answer_with_context(llm, docs, question, history=history)
        text = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
        return RagResult(durum="basarili", yanit=text, kaynaklar=docs)
    except Exception:
        logger.exception("answer_with_context (LLM çağrısı) başarısız oldu")
        # İç hata detayını dışarı sızdırma.
        return RagResult(
            durum="hata",
            yanit="Yanıt üretilirken bir sorun oluştu. Lütfen daha sonra tekrar deneyin.",
            kaynaklar=docs,
        )