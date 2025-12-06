import os
import torch

# LangChain ve Qdrant Kütüphaneleri
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. AYARLAR
# ==========================================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "Tubitak_Dokumanlar"
EMBEDDING_MODEL_NAME = "ytu-ce-cosmos/turkish-e5-large"
LLM_MODEL_NAME = "gemma3:12b"   #gemma3:12b qwen2.5:14b

# Cihaz Seçimi (CUDA/CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Çalışma Modu: {device.upper()}")

# ==========================================
# 2. BAĞLANTILAR (Client, Embedding, LLM)
# ==========================================

# Qdrant Client
try:
    client = QdrantClient(url=QDRANT_URL)
    # Bağlantıyı test et
    client.get_collections()
    print("✅ Qdrant bağlantısı başarılı.")
except Exception as e:
    print(f"❌ Qdrant'a bağlanılamadı: {e}")
    print("Docker konteynerinin çalıştığından emin olun.")
    exit()

# Embedding Modeli
print("🧠 Embedding modeli yükleniyor...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)

# LLM (Gemma 3)
print("🤖 LLM (Gemma) hazırlanıyor...")
llm = OllamaLLM(
    model=LLM_MODEL_NAME,
    temperature=0.1,
    top_p=0.9,
    repeat_penalty=1.1,
    num_predict=1024,
)


# ==========================================
# 3. YARDIMCI FONKSİYONLAR
# ==========================================

def get_vector_store():
    """LangChain uyumlu VectorStore nesnesini döndürür"""
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


def get_context_and_print(query: str, doc_type: str = None):
    """
    Veritabanından bilgiyi çeker ve EKRANA YAZDIRIR.
    """
    vector_store = get_vector_store()

    # Filtreleme Mantığı
    search_kwargs = {"k": 3}  # En alakalı 3 parçayı getir

    if doc_type:
        print(f"ℹ️ Filtre uygulanıyor: {doc_type}")
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.file_type",
                    match=models.MatchValue(value=doc_type)
                )
            ]
        )
        search_kwargs["filter"] = filter_condition

    # Semantik Arama Yap
    docs_with_scores = vector_store.similarity_search_with_score(
        query,
        **search_kwargs
    )

    # Gelen içerikleri birleştir ve YAZDIR
    context_parts = []

    print("\n" + "=" * 50)
    print("🔍 VEKTÖR VERİTABANINDAN GETİRİLEN CHUNK'LAR")
    print("=" * 50)

    if not docs_with_scores:
        print("❌ Hiçbir eşleşme bulunamadı!")
        return None

    for i, (doc, score) in enumerate(docs_with_scores, 1):
        source = doc.metadata.get("source", "Bilinmiyor")
        sheet = doc.metadata.get("sheet", "-")

        print(f"\n📄 [CHUNK {i}] (Benzerlik Skoru: {score:.4f})")
        print(f"   📂 Kaynak: {source}")
        if sheet != "-": print(f"   📑 Sayfa: {sheet}")
        print("-" * 30)
        print(f"{doc.page_content}")
        print("-" * 30)

        context_parts.append(doc.page_content)

    return "\n\n---\n\n".join(context_parts)


# ==========================================
# 4. ANA ÇALIŞTIRMA FONKSİYONU
# ==========================================

def run_rag_pipeline(question: str, doc_type: str = None):
    print(f"\n📥 KULLANICI SORUSU: {question}")

    # 1. Chunkları getir ve yazdır
    context_text = get_context_and_print(question, doc_type)

    if not context_text:
        print("⚠️ Yeterli bilgi bulunamadığı için model çalıştırılmadı.")
        return

    # 2. Prompt Hazırla
    prompt_template = """Sen yardımcı bir yapay zeka asistanısın. Aşağıdaki bağlam bilgisini kullanarak kullanıcının sorusunu cevapla.
    Eğer bağlamda cevabı bulamazsan, uydurma, sadece "Bilgim yok" de.

    Bağlam (Veritabanından Gelen Bilgi):
    {context}

    Kullanıcı Sorusu:
    {question}

    Cevap:"""

    final_prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = final_prompt | llm

    # 3. Modele Gönder
    print("\n" + "=" * 50)
    print("🤖 GEMMA MODELİ DÜŞÜNÜYOR...")
    print("=" * 50)

    response = chain.invoke({
        "context": context_text,
        "question": question
    })

    # 4. Cevabı Yazdır
    print(f"\n{response}\n")
    print("=" * 50)


# ==========================================
# 5. TEST ALANI (Burayı Değiştirip Çalıştır)
# ==========================================
if __name__ == "__main__":
    # BURAYA İSTEDİĞİN SORUYU YAZ
    soru = "Disiplin kurulu üyeleri kimlerden oluşur ve kim tarafından seçilir?"

    # doc_type="excel" diyerek sadece excelde aratabilirsin,
    # veya None diyerek hepsinde aratabilirsin.

    run_rag_pipeline(soru, doc_type="pdf")