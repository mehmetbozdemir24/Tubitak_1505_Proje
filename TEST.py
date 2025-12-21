import os
import torch
import getpass

# LangChain ve Qdrant Kütüphaneleri
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Google Gemini Kütüphanesi
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# 1. AYARLAR
# ==========================================

# --- SEÇİM YAPIN ---
# "ollama" veya "gemini" yazarak motoru değiştirin.
LLM_PROVIDER = "ollama"  # veya "gemini"

# Qdrant ve Embedding Ayarları
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "Tubitak_Dokumanlar"
EMBEDDING_MODEL_NAME = "ytu-ce-cosmos/turkish-e5-large"

# Model İsimleri
OLLAMA_MODEL_NAME = "gemma3:12b"
GEMINI_MODEL_NAME = "gemini-2.5-flash"  # veya "gemini-1.5-pro"

# Cihaz Seçimi (CUDA/CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Çalışma Modu: {device.upper()}")

# ==========================================
# 2. BAĞLANTILAR (Client, Embedding)
# ==========================================

# Qdrant Client
try:
    client = QdrantClient(url=QDRANT_URL)
    client.get_collections()
    print("✅ Qdrant bağlantısı başarılı.")
except Exception as e:
    print(f"❌ Qdrant'a bağlanılamadı: {e}")
    exit()

# Embedding Modeli (Her iki LLM için de ortak)
print("🧠 Embedding modeli yükleniyor...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)

# ==========================================
# 3. LLM KURULUMU (Ollama vs Gemini)
# ==========================================

llm = None

if LLM_PROVIDER == "gemini":
    print(f"🤖 LLM Modu: GOOGLE GEMINI ({GEMINI_MODEL_NAME}) hazırlanıyor...")

    os.environ["GOOGLE_API_KEY"] = ""

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        temperature=0.1,
        max_retries=2,
    )

elif LLM_PROVIDER == "ollama":
    print(f"🤖 LLM Modu: LOCAL OLLAMA ({OLLAMA_MODEL_NAME}) hazırlanıyor...")
    llm = OllamaLLM(
        model=OLLAMA_MODEL_NAME,
        temperature=0.1,
        top_p=0.9,
        repeat_penalty=1.1,
        num_predict=1024,
    )

else:
    raise ValueError("Geçersiz LLM_PROVIDER seçimi! 'ollama' veya 'gemini' olmalı.")


# ==========================================
# 4. YARDIMCI FONKSİYONLAR (RAG)
# ==========================================

def get_vector_store():
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


def get_context_and_print(query: str, permission: str, doc_type: str = None, k: int = 3, SCORE_THRESHOLD=0.50):
    vector_store = get_vector_store()

    # Filtreler
    search_kwargs = {"k": k}
    must_conditions = []

    must_conditions.append(
        models.FieldCondition(
            key="metadata.permission",
            match=models.MatchValue(value=permission)
        )
    )

    if doc_type:
        must_conditions.append(
            models.FieldCondition(
                key="metadata.file_type",
                match=models.MatchValue(value=doc_type)
            )
        )

    if must_conditions:
        search_kwargs["filter"] = models.Filter(must=must_conditions)

    # Arama Yap
    docs_with_scores = vector_store.similarity_search_with_score(
        query,
        **search_kwargs
    )

    context_parts = []

    print("\n" + "=" * 50)
    print("🔍 VEKTÖR SONUÇLARI ANALİZİ")
    print("=" * 50)

    filtered_docs = []
    for doc, score in docs_with_scores:
        if score >= SCORE_THRESHOLD:
            filtered_docs.append((doc, score))
        else:
            print(f"⚠️ ELENDİ (Düşük Skor: {score:.4f}) - {doc.metadata.get('source')}")

    if not filtered_docs:
        print("❌ Yeterince benzer sonuç bulunamadı (Eşik altı).")
        return None

    for i, (doc, score) in enumerate(filtered_docs, 1):
        print(f"\n📄 [CHUNK {i}] (Benzerlik Skoru: {score:.4f})")
        print(f"   📂 Kaynak: {doc.metadata.get('source')}")
        print(f"   🔒 Yetki: {doc.metadata.get('permission')}")
        print("-" * 30)
        print(f"{doc.page_content}")
        print("-" * 30)
        context_parts.append(doc.page_content)

    return "\n\n---\n\n".join(context_parts)


# ==========================================
# 5. ANA ÇALIŞTIRMA FONKSİYONU
# ==========================================
import time
def run_rag_pipeline(question: str, permission: str, doc_type: str = None, k: int = 3, SCORE_THRESHOLD=0.50):
    print(f"\n📥 KULLANICI SORUSU: {question}")

    # 1. Chunkları getir BURASI
    baslangic = time.perf_counter()
    context_text = get_context_and_print(question, permission, doc_type, k, SCORE_THRESHOLD)

    bitis = time.perf_counter()
    gecen_sure_ms = (bitis - baslangic) * 1000

    # ".2f" ile virgülden sonra sadece 2 basamak gösteririz
    print(f"İşlem süresi: {gecen_sure_ms:.2f} ms")

    # Context yoksa iptal et
    if not context_text:
        print("\n" + "=" * 50)
        print("🤖 SİSTEM CEVABI")
        print("=" * 50)
        print("\nBilgim yok.\n")
        print("=" * 50)
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
    print(f"🤖 {LLM_PROVIDER.upper()} MODELİ DÜŞÜNÜYOR...")
    print("=" * 50)

    response = chain.invoke({
        "context": context_text,
        "question": question
    })

    # 4. Cevabı Yazdır
    # Gemini bazen "content" objesi dönebilir, Ollama string döner. LangChain bunu genelde yönetir ama
    # garanti olsun diye string'e çevirelim veya doğrudan yazdıralım.

    final_response = response.content if hasattr(response, 'content') else response

    print(f"\n{final_response}\n")
    print("=" * 50)


# ==========================================
# 6. TEST ALANI
# ==========================================

if __name__ == "__main__":

    # Sayacı başlat

    soru = "BAP komisyonu kimlerden oluşur?"

    run_rag_pipeline(
        soru,
        permission="manager",
        doc_type="pdf",
        k=5,
        SCORE_THRESHOLD=0.45
    )