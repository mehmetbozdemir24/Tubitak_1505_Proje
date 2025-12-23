"""
Chunking Modülü Kullanım Örneği

Bu script, chunking modülünün temel kullanımını gösterir.
Farklı dosya tiplerini (PDF, DOCX, PPTX, XLSX) yükleyip chunk'lara ayırır.
"""

import sys
import os

# Proje kök dizinini Python path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.chunking import ChunkingProcessor


def main():
    """Ana fonksiyon - Chunking işlemlerini gösterir"""
    
    print("=" * 60)
    print("TÜBİTAK 1505 Projesi - Chunking Modülü Örneği")
    print("=" * 60)
    print()
    
    # ChunkingProcessor oluştur
    processor = ChunkingProcessor(
        chunk_size=1000,      # Her chunk maksimum 1000 karakter
        chunk_overlap=200     # Chunk'lar arası 200 karakter örtüşme
    )
    
    print("Chunking Processor Ayarları:")
    print(f"  - Chunk Boyutu: {processor.chunk_size} karakter")
    print(f"  - Chunk Örtüşme: {processor.chunk_overlap} karakter")
    print()
    
    # Örnek: Tek bir dosya işleme
    print("Örnek 1: Tek Dosya İşleme")
    print("-" * 60)
    
    # Not: Bu örnekte gerçek bir dosya olmadığı için try-except kullanıyoruz
    example_file = "path/to/example.pdf"
    print(f"Dosya: {example_file}")
    
    try:
        chunks = processor.process_file(example_file)
        print(f"✅ Başarılı! {len(chunks)} chunk oluşturuldu")
        
        # İlk chunk'ı göster
        if chunks:
            print(f"\nİlk Chunk Örneği:")
            print(f"  İçerik: {chunks[0].page_content[:100]}...")
            print(f"  Metadata: {chunks[0].metadata}")
    except FileNotFoundError:
        print("⚠️  Örnek dosya bulunamadı (bu normal, sadece örnek)")
    except Exception as e:
        print(f"❌ Hata: {e}")
    
    print()
    
    # Örnek: Birden fazla dosya işleme
    print("Örnek 2: Çoklu Dosya İşleme")
    print("-" * 60)
    
    example_files = [
        "path/to/file1.pdf",
        "path/to/file2.docx",
        "path/to/file3.pptx",
        "path/to/file4.xlsx"
    ]
    
    print("Dosyalar:")
    for f in example_files:
        print(f"  - {f}")
    print()
    
    try:
        all_chunks = processor.process_multiple_files(example_files)
        print(f"\n✅ Toplam {len(all_chunks)} chunk oluşturuldu")
        
        # İstatistikleri göster
        stats = processor.get_chunk_statistics(all_chunks)
        print("\nChunk İstatistikleri:")
        print(f"  - Toplam Chunk: {stats['total_chunks']}")
        print(f"  - Ortalama Boyut: {stats['avg_chunk_size']:.0f} karakter")
        print(f"  - Min Boyut: {stats['min_chunk_size']} karakter")
        print(f"  - Max Boyut: {stats['max_chunk_size']} karakter")
        print(f"  - Toplam Karakter: {stats['total_characters']}")
        
    except Exception as e:
        print(f"⚠️  Örnek dosyalar bulunamadı (bu normal, sadece örnek)")
    
    print()
    print("=" * 60)
    print("Kullanım Notu:")
    print("Gerçek dosyalarla çalışmak için:")
    print("1. Dosya yollarını güncelleyin")
    print("2. Chunk boyutu ve örtüşme parametrelerini ihtiyaca göre ayarlayın")
    print("3. processor.process_file() veya process_multiple_files() kullanın")
    print("=" * 60)


if __name__ == "__main__":
    main()
