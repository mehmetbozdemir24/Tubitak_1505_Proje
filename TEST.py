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
LLM_MODEL_NAME = "gemma3:12b"

# Cihaz Seçimi (CUDA/CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Çalışma Modu: {device.upper()}")

# ==========================================
# 2. BAĞLANTILAR (Client, Embedding, LLM)
# ==========================================

try:
    client = QdrantClient(url=QDRANT_URL)
    client.get_collections()
    print("✅ Qdrant bağlantısı başarılı.")
except Exception as e:
    print(f"❌ Qdrant'a bağlanılamadı: {e}")
    exit()

print("🧠 Embedding modeli yükleniyor...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)

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
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

# GÜNCELLEME 1: k parametresi eklendi (varsayılan 3)
def get_context_and_print(query: str, permission: str, doc_type: str = None, k: int = 3):
    vector_store = get_vector_store()

    # 1. FİLTRELERİ HAZIRLA
    # GÜNCELLEME: k parametresi buraya bağlandı
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

    docs_with_scores = vector_store.similarity_search_with_score(
        query,
        **search_kwargs
    )

    context_parts = []

    # Eşik Değeri
    SCORE_THRESHOLD = 0.60

    print("\n" + "=" * 50)
    print("🔍 VEKTÖR SONUÇLARI ANALİZİ")
    print("=" * 50)

    filtered_docs = []
    for doc, score in docs_with_scores:
        if score >= SCORE_THRESHOLD:
            filtered_docs.append((doc, score))
        else:
            print(f"⚠️ ELENDİ (Düşük Skor: {score:.4f}) - {doc.metadata.get('source')}")

    # Eğer hiç belge kalmadıysa None dönüyoruz
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
# 4. ANA ÇALIŞTIRMA FONKSİYONU
# ==========================================

# GÜNCELLEME 2: k parametresi buraya da eklendi
def run_rag_pipeline(question: str, permission: str, doc_type: str = None, k: int = 3):
    print(f"\n📥 KULLANICI SORUSU: {question}")

    # 1. Chunkları getir
    context_text = get_context_and_print(question, permission, doc_type, k)

    # GÜNCELLEME 3: Eğer context yoksa (eşik altındaysa), doğrudan "Bilgim yok" de.
    if not context_text:
        print("\n" + "=" * 50)
        print("🤖 SİSTEM CEVABI (Model Çalıştırılmadı)")
        print("=" * 50)
        print("\nBilgim yok.\n") # Kullanıcının göreceği cevap
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
# 5. TEST ALANI
# ==========================================
if __name__ == "__main__":

    #soru = "Disiplin kurulu üyeleri kimlerden oluşur ve kim tarafından seçilir?"
    soru = "Disiplin  suç ve cezaları nelerdir?"

    run_rag_pipeline(soru, permission="user", doc_type="pdf", k=5)