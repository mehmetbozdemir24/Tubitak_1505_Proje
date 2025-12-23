"""
Chunking Modülü Demo

Bu script, chunking modülünün temel özelliklerini örnek verilerle gösterir.
Metin dokümanları üzerinde farklı chunking stratejilerini uygular ve sonuçları gösterir.
"""

import sys
import os

# Proje kök dizinini Python path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.chunking import ChunkingProcessor, DocumentLoader
from langchain_core.documents import Document


def create_sample_text_document():
    """
    TÜBİTAK 1505 projesi hakkında örnek bir metin dokümanı oluşturur.
    
    Bu fonksiyon, demo amaçlı kullanılmak üzere proje mimarisini açıklayan
    bir örnek doküman oluşturur. Gerçek bir dosya okumadan chunking işlemlerini
    test etmek için kullanılır.
    
    Returns:
        Document: Örnek içerik ve metadata içeren bir Document objesi
    """
    content = """
TÜBİTAK 1505 Projesi - Kurumsal Doküman Danışmanı

Bu proje, kurumsal dokümanlardan çıkarılan bilgiler üzerinde anlam-temelli aramalar 
yapan ve akıllı yanıtlar üreten bir yapay zekâ asistanıdır. Sistem, modern derin 
öğrenme modelleri ve vektör veritabanı teknolojileriyle hızlı, doğru ve bağlamsal 
yanıtlar sağlar.

Sistem Mimarisi:

1. Doküman Ön İşleme & Chunking
   - Proje kapsamındaki dokümanların okunması ve temizlenmesi
   - Metinlerin anlamlı parçalara (chunks) bölünmesi
   - Görevliler: Engin, Batuhan

2. Embedding Vektörleştirme
   - Cosmos-e5-large ile metin parçalarının vektörleştirilmesi
   - Anlamsal temsil oluşturulması
   - Görevliler: Mehmet, Hasan

3. Vektör Veritabanı Yönetimi
   - Embedding'lerin Docker üzerinden Qdrant'a yüklenmesi
   - Hızlı ve ölçeklenebilir arama altyapısının sağlanması
   - Görevliler: Süleyman, Eren

4. Akıllı Yanıt Üretimi
   - Qdrant'tan bağlama uygun sonuçların çağrılması
   - Gemma3-12B ve Qwen3-8B ile doğal dil yanıtlarının üretilmesi
   - Görevliler: Hasan, Eren

Teknoloji Stack:
- Embedding Modeli: Cosmos-e5-large
- Vektör Veritabanı: Qdrant (Docker)
- LLM Modelleri: Gemma3-12B, Qwen3-8B
- Container: Docker

Bu sistem, RAG (Retrieval-Augmented Generation) mimarisine dayanır ve kurumsal 
dokümanlar üzerinde akıllı bir sorgulama katmanı oluşturur. Kullanıcılar, doğal 
dilde soru sorabilir ve sistem, ilgili dokümanlardan çıkarılan bilgilere dayanarak 
akıllı yanıtlar üretir.
"""
    return Document(page_content=content.strip(), metadata={"source": "sample_text", "type": "demo"})


def demo_basic_chunking():
    """Temel chunking işlemini gösterir"""
    print("\n" + "=" * 70)
    print("Demo 1: Temel Chunking İşlemi")
    print("=" * 70)
    
    # Örnek doküman oluştur
    doc = create_sample_text_document()
    
    print(f"\nOrijinal Doküman:")
    print(f"  - Toplam karakter: {len(doc.page_content)}")
    print(f"  - Kaynak: {doc.metadata['source']}")
    
    # ChunkingProcessor oluştur
    processor = ChunkingProcessor(
        chunk_size=300,   # Küçük chunk boyutu (demo için)
        chunk_overlap=50  # 50 karakter örtüşme
    )
    
    # Text splitter ile chunk'la
    chunks = processor.text_splitter.split_documents([doc])
    
    print(f"\nChunking Sonuçları:")
    print(f"  - Chunk sayısı: {len(chunks)}")
    print(f"  - Chunk boyutu: {processor.chunk_size} karakter (maks)")
    print(f"  - Örtüşme: {processor.chunk_overlap} karakter")
    
    # Her chunk'ı göster
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Boyut: {len(chunk.page_content)} karakter")
        print(f"İçerik önizleme: {chunk.page_content[:100]}...")


def demo_chunk_statistics():
    """Chunk istatistiklerini gösterir"""
    print("\n" + "=" * 70)
    print("Demo 2: Chunk İstatistikleri")
    print("=" * 70)
    
    # Örnek doküman oluştur
    doc = create_sample_text_document()
    
    # Farklı chunk boyutlarını dene
    chunk_sizes = [200, 500, 1000]
    
    for size in chunk_sizes:
        processor = ChunkingProcessor(
            chunk_size=size,
            chunk_overlap=50
        )
        
        chunks = processor.text_splitter.split_documents([doc])
        stats = processor.get_chunk_statistics(chunks)
        
        print(f"\nChunk Boyutu: {size}")
        print(f"  - Toplam chunk: {stats['total_chunks']}")
        print(f"  - Ort. boyut: {stats['avg_chunk_size']:.0f} karakter")
        print(f"  - Min boyut: {stats['min_chunk_size']} karakter")
        print(f"  - Max boyut: {stats['max_chunk_size']} karakter")


def demo_custom_separators():
    """Özel ayırıcılarla chunking gösterir"""
    print("\n" + "=" * 70)
    print("Demo 3: Özel Ayırıcılarla Chunking")
    print("=" * 70)
    
    doc = create_sample_text_document()
    
    # Sadece paragraf bazlı ayırma
    processor1 = ChunkingProcessor(
        chunk_size=500,
        chunk_overlap=0,
        separators=["\n\n", "\n"]  # Sadece paragraf ve satır
    )
    
    chunks1 = processor1.text_splitter.split_documents([doc])
    
    print(f"\nParagraf Bazlı Ayırma:")
    print(f"  - Chunk sayısı: {len(chunks1)}")
    print(f"  - Ayırıcılar: [\\n\\n, \\n]")
    
    # Nokta bazlı ayırma
    processor2 = ChunkingProcessor(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]  # Paragraf, satır, nokta, boşluk
    )
    
    chunks2 = processor2.text_splitter.split_documents([doc])
    
    print(f"\nCümle Bazlı Ayırma:")
    print(f"  - Chunk sayısı: {len(chunks2)}")
    print(f"  - Ayırıcılar: [\\n\\n, \\n, '. ', ' ']")


def demo_metadata_preservation():
    """Metadata korunmasını gösterir"""
    print("\n" + "=" * 70)
    print("Demo 4: Metadata Koruma")
    print("=" * 70)
    
    # Metadata'lı doküman oluştur
    doc = Document(
        page_content="Python programlama dili. " * 50,
        metadata={
            "source": "python_guide.pdf",
            "author": "John Doe",
            "page": 5,
            "chapter": "Giriş"
        }
    )
    
    processor = ChunkingProcessor(chunk_size=200, chunk_overlap=20)
    chunks = processor.text_splitter.split_documents([doc])
    
    print(f"\nOrijinal Metadata:")
    for key, value in doc.metadata.items():
        print(f"  - {key}: {value}")
    
    print(f"\nChunk'larda Metadata:")
    print(f"  - Toplam chunk: {len(chunks)}")
    
    # İlk chunk'ın metadata'sını göster
    if chunks:
        print(f"\nİlk Chunk Metadata:")
        for key, value in chunks[0].metadata.items():
            print(f"  - {key}: {value}")


def main():
    """Ana fonksiyon - Tüm demoları çalıştırır"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "TÜBİTAK 1505 - Chunking Modülü Demo" + " " * 18 + "║")
    print("╚" + "═" * 68 + "╝")
    
    try:
        demo_basic_chunking()
        demo_chunk_statistics()
        demo_custom_separators()
        demo_metadata_preservation()
        
        print("\n" + "=" * 70)
        print("Demo tamamlandı! ✅")
        print("=" * 70)
        print("\nÖnemli Notlar:")
        print("  1. Chunk boyutu, belge türüne göre ayarlanmalıdır")
        print("  2. Örtüşme, bağlamsal bütünlüğü korumak için önemlidir")
        print("  3. Ayırıcılar, metin yapısına göre özelleştirilebilir")
        print("  4. Metadata, vektör veritabanında filtreleme için kullanılır")
        print()
        
    except ImportError as e:
        print(f"\n❌ Import Hatası: {e}")
        print("Lütfen gerekli kütüphaneleri yükleyin: pip install -r requirements.txt")
    except Exception as e:
        print(f"\n❌ Beklenmeyen Hata: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
