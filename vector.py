import os
import pickle
import json
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

# --- AYARLAR ---
INPUT_FOLDER = "document_chunks"
OUTPUT_FOLDER = "vector_store"
MODEL_ID = "ytu-ce-cosmos/turkish-e5-large"

# Klasör kontrolü
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# 1. GPU ve Model Hazırlığı
print("Model Yükleniyor")
device = "cuda" if torch.cuda.is_available() else "cpu"

hf_embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_ID,
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

def extract_text_from_chunk(chunk):
    """Chunk verisinin tipine göre metni ayıklar."""
    if isinstance(chunk, str):
        return chunk
    elif hasattr(chunk, 'page_content'): # LangChain Document objesi
        return chunk.page_content
    elif isinstance(chunk, dict): 
        for key in ['page_content', 'text', 'content', 'body']:
            if key in chunk:
                return chunk[key]
        # Eğer key bulunamazsa mecbur string'e çevir
        return str(chunk)
    else:
        return str(chunk)

def process_file_pair(pkl_path, json_path, output_path):
    # Verileri Oku
    try:
        with open(pkl_path, 'rb') as f:
            chunks_data = pickle.load(f)
        with open(json_path, 'r', encoding='utf-8') as f:
            metadatas = json.load(f)
    except Exception as e:
        print(f"Dosya okuma hatası ({pkl_path}): {e}")
        return False

    # Chunk listesini kontrol et ve metinleri ayıkla
    texts = []
    if isinstance(chunks_data, list) and len(chunks_data) > 0:
        # Her bir chunk için özel fonksiyonu çağır
        texts = [extract_text_from_chunk(chunk) for chunk in chunks_data]
    else:
        print(f"Boş veya geçersiz dosya: {pkl_path}")
        return False # Boş dosya

    # Sayı kontrolü
    if len(texts) != len(metadatas):
        print(f"SATIR SAYISI UYUŞMAZLIĞI: {pkl_path}")
        print(f"   -> Metin: {len(texts)}, Metadata: {len(metadatas)}")

    # Vektörleştirme
    # Boş metinleri temizle (Artık texts içinde kesinlikle string var)
    safe_texts = []
    for t in texts:
        if t and isinstance(t, str) and t.strip():
            safe_texts.append(t)
        else:
            safe_texts.append(" ")
    
    try:
        vectors = hf_embeddings.embed_documents(safe_texts)
    except Exception as e:
        print(f"Embedding Hatası ({pkl_path}): {e}")
        return False

    # Birleştir ve Kaydet
    final_data = []
    for text, meta, vec in zip(texts, metadatas, vectors):
        final_data.append({
            "text": text,
            "metadata": meta,
            "vector": vec
        })

    with open(output_path, "wb") as f:
        pickle.dump(final_data, f)
    
    return True

# 2. Döngü
files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.pkl')]
print(f"{len(files)} adet dosya işlenecek.")

count = 0
for pkl_file in tqdm(files, desc="Vektörleştiriliyor"):
    base_name = pkl_file.replace(".pkl", "")
    json_file = f"{base_name}_metadata.json"
    
    pkl_path = os.path.join(INPUT_FOLDER, pkl_file)
    json_path = os.path.join(INPUT_FOLDER, json_file)
    output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_embedded.pkl")
    
    if os.path.exists(json_path):
        if process_file_pair(pkl_path, json_path, output_path):
            count += 1
    # else:
    #     print(f"Meta dosyası yok: {json_file}")

print(f"\n İşlem Tamamlandı! {count} dosya 'vector_store' klasörüne kaydedildi.")