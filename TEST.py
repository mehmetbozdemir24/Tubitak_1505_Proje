import os
import torch
import time
from hallucination_validator import HallucinationValidator

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

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


def get_context_and_print(query: str, permission: str, doc_type: str = None, k: int = 3, SCORE_THRESHOLD=0.70):
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
        return None, []

    docs_list = []
    for i, (doc, score) in enumerate(filtered_docs, 1):
        doc.metadata["score"] = score
        docs_list.append(doc)
        print(f"\n📄 [CHUNK {i}] (Benzerlik Skoru: {score:.4f})")
        print(f"   📂 Kaynak: {doc.metadata.get('source')}")
        print(f"   🔒 Yetki: {doc.metadata.get('permission')}")
        print("-" * 30)
        print(f"{doc.page_content}")
        print("-" * 30)
        context_parts.append(doc.page_content)

    return "\n\n---\n\n".join(context_parts), docs_list


# ==========================================
# 5. ANA ÇALIŞTIRMA FONKSİYONU
# ==========================================
import time
def run_rag_pipeline(question: str, permission: str, doc_type: str = None, k: int = 3, SCORE_THRESHOLD=0.70):
    print(f"\n📥 KULLANICI SORUSU: {question}")

    # 1. Chunkları getir BURASI
    baslangic = time.perf_counter()
    context_text, retrieved_docs = get_context_and_print(question, permission, doc_type, k, SCORE_THRESHOLD)

    bitis = time.perf_counter()
    gecen_sure_ms = (bitis - baslangic) * 1000

    print(f"İşlem süresi: {gecen_sure_ms:.2f} ms")

    if not context_text:
        print("\n" + "=" * 50)
        print("🤖 SİSTEM CEVABI")
        print("=" * 50)
        print("\nBu konuda belgelerimde yeterli kalitede bilgi bulunamadı.\n")
        print("=" * 50)
        return

    # 2. Prompt Hazırla
    prompt_template = """Sen yardımcı bir yapay zeka asistanısın. Aşağıdaki bağlam bilgisini kullanarak kullanıcının sorusunu cevapla.
    Eğer bağlamda cevabı bulamazsan, uydurma, sadece "Bilgim yok" de.

    Bağlam (Veritabanından Gelen Bilgi):
    {context}

    Kullanıcı Sorusu:
    {question}

    KATÎ KURALLAR:
    1. SADECE yukarıdaki bağlamdan cevap ver.
    2. Bağlamda cevap YOKSA: "Bu konuda belgelerimde bilgi bulunmuyor" de.
    3. ASLA tahmin yapma veya kendi bilgini kullanma.
    4. Belirsizlik varsa açıkça belirt.

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

    is_valid, validation_msg = HallucinationValidator.validate_response(
        question, final_response, retrieved_docs
    )
    
    if not is_valid:
        print(f"\n⚠️ HALÜSINASYON UYARISI: {validation_msg}")
        print("Sistem güvenlik nedeniyle yanıtı reddetdi.\n")
        print("=" * 50)
        return

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
        SCORE_THRESHOLD=0.70
    )