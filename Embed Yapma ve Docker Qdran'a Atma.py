import os
import pickle
import torch
from uuid import uuid4

# Gerekli Kütüphaneler
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# 1. AYARLAR VE TANIMLAMALAR
# ==========================================
PKL_PATH = "Tubitak_1505_Proje/tum_dokumanlar_final_last.pkl"  # Senin dosyanın yolu
COLLECTION_NAME = "Tubitak_Dokumanlar"  # Qdrant içindeki koleksiyon adı
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL_NAME = "ytu-ce-cosmos/turkish-e5-large"  # Türkçe için en iyi modellerden biri

# ==========================================
# 2. PKL DOSYASINI YÜKLEME
# ==========================================
print(f"📂 '{PKL_PATH}' dosyası yükleniyor...")

try:
    if not os.path.exists(PKL_PATH):
        # Eğer full path bulamazsa scriptin olduğu yerdeki ismi dener
        PKL_PATH = "tum_dokumanlar_final_last.pkl"

    with open(PKL_PATH, "rb") as f:
        all_documents = pickle.load(f)
    print(f"✅ {len(all_documents)} adet belge (chunk) başarıyla yüklendi.")

except FileNotFoundError:
    print(f"❌ HATA: '{PKL_PATH}' dosyası bulunamadı! Lütfen dosya yolunu kontrol et.")
    exit()

# ==========================================
# 3. EMBEDDING MODELİNİ HAZIRLAMA
# ==========================================
print("🧠 Embedding modeli hazırlanıyor (GPU/CPU kontrol ediliyor)...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Çalışma Modu: {device.upper()}")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}  # Cosine similarity için önemli
)

# ==========================================
# 4. QDRANT BAĞLANTISI VE KOLEKSİYON KONTROLÜ
# ==========================================
print(f"🔌 Qdrant'a bağlanılıyor: {QDRANT_URL}")
client = QdrantClient(url=QDRANT_URL)

# Koleksiyon var mı diye kontrol et
if not client.collection_exists(COLLECTION_NAME):
    print(f"⚠️ Koleksiyon '{COLLECTION_NAME}' bulunamadı. Yeni oluşturuluyor...")

    # ytu-ce-cosmos/turkish-e5-large modeli 1024 boyutlu vektör üretir.
    # Bu yüzden size=1024 olmak zorunda.
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    print(f"✅ Koleksiyon oluşturuldu.")
else:
    print(f"ℹ️ Koleksiyon '{COLLECTION_NAME}' zaten mevcut. Veriler üzerine eklenecek.")

# ==========================================
# 5. VEKTÖRLEŞTİRME VE KAYDETME
# ==========================================
print("🚀 Vektörleştirme ve veritabanına yükleme işlemi başlıyor...")

# Langchain Qdrant entegrasyonunu başlat
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# Belgeleri Qdrant'a ekle
# uuid4 kullanarak her belgeye benzersiz bir ID veriyoruz
chunk_ids = [str(uuid4()) for _ in all_documents]

try:
    vector_store.add_documents(documents=all_documents, ids=chunk_ids)
    print(f"🎉 İŞLEM TAMAMLANDI! {len(all_documents)} adet chunk başarıyla Qdrant'a yüklendi.")
    print("Artık bu veriler üzerinde semantik arama yapabilirsiniz.")

except Exception as e:
    print(f"❌ Yükleme sırasında hata oluştu: {e}")