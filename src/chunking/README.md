# Chunking Modülü

TÜBİTAK 1505 Projesi için geliştirilmiş doküman işleme ve parçalama (chunking) modülü.

## 📋 Genel Bakış

Bu modül, farklı dosya formatlarından (PDF, DOCX, PPTX, XLSX) metin çıkarma ve bunları anlamlı parçalara (chunks) ayırma işlemlerini gerçekleştirir. LangChain kütüphanesi kullanılarak geliştirilmiştir.

## 🎯 Özellikler

- ✅ Çoklu dosya formatı desteği (PDF, DOCX, PPTX, XLSX)
- ✅ Yapılandırılabilir chunk boyutu ve örtüşme
- ✅ Anlamsal bütünlüğü koruyan akıllı metin bölme
- ✅ Metadata yönetimi (dosya kaynağı, chunk ID, boyut bilgisi)
- ✅ İstatistik ve analiz araçları
- ✅ Toplu dosya işleme
- ✅ Kapsamlı hata yönetimi

## 📦 Kurulum

Gerekli bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

## 🚀 Kullanım

### Temel Kullanım

```python
from src.chunking import ChunkingProcessor

# Processor oluştur
processor = ChunkingProcessor(
    chunk_size=1000,      # Her chunk maksimum 1000 karakter
    chunk_overlap=200     # Chunk'lar arası 200 karakter örtüşme
)

# Tek bir dosyayı işle
chunks = processor.process_file('path/to/document.pdf')

# Sonuçları incele
print(f"Toplam {len(chunks)} chunk oluşturuldu")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk.page_content)} karakter")
    print(f"Metadata: {chunk.metadata}")
```

### Çoklu Dosya İşleme

```python
from src.chunking import ChunkingProcessor

processor = ChunkingProcessor()

# Birden fazla dosyayı işle
file_paths = [
    'documents/file1.pdf',
    'documents/file2.docx',
    'documents/file3.pptx',
    'documents/file4.xlsx'
]

all_chunks = processor.process_multiple_files(file_paths)

# İstatistikleri al
stats = processor.get_chunk_statistics(all_chunks)
print(f"Toplam chunk: {stats['total_chunks']}")
print(f"Ortalama boyut: {stats['avg_chunk_size']:.0f} karakter")
```

### Sadece Doküman Yükleme

```python
from src.chunking import DocumentLoader

loader = DocumentLoader()

# Desteklenen formatları öğren
formats = loader.get_supported_formats()
print(f"Desteklenen formatlar: {formats}")

# Bir dokümanı yükle (chunk'lamadan)
documents = loader.load_document('path/to/file.pdf')
```

## 🔧 Yapılandırma

### ChunkingProcessor Parametreleri

- **chunk_size** (int, varsayılan: 1000): Her bir chunk'ın maksimum karakter sayısı
- **chunk_overlap** (int, varsayılan: 200): Chunk'lar arasındaki örtüşme miktarı
- **separators** (List[str], opsiyonel): Metni bölmek için kullanılacak ayırıcılar

### Ayırıcılar

Varsayılan ayırıcılar (öncelik sırasına göre):
1. `\n\n` - Paragraf sonu
2. `\n` - Satır sonu
3. `. ` - Nokta ve boşluk
4. ` ` - Boşluk
5. `""` - Karakter seviyesi

Özel ayırıcılar tanımlayabilirsiniz:

```python
processor = ChunkingProcessor(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", ",", " "]
)
```

## 📊 Metadata

Her chunk aşağıdaki metadata'yı içerir:

- **chunk_id**: Chunk'ın sıra numarası
- **chunk_size**: Chunk'ın karakter sayısı
- **source_file**: Kaynak dosya yolu
- Orijinal dosyadan gelen diğer metadata (sayfa numarası, vb.)

## 🧪 Test

Testleri çalıştırmak için:

```bash
# Tüm testleri çalıştır
python -m unittest discover tests/chunking -v

# Belirli bir test dosyasını çalıştır
python -m unittest tests.chunking.test_document_loader -v
python -m unittest tests.chunking.test_chunking_processor -v
```

## 📝 Örnek

Detaylı kullanım örneği için:

```bash
python examples/chunking_example.py
```

## 🏗️ Modül Yapısı

```
src/chunking/
├── __init__.py                 # Modül giriş noktası
├── document_loader.py          # Dosya yükleme sınıfı
└── chunking_processor.py       # Ana işleme sınıfı

tests/chunking/
├── __init__.py
├── test_document_loader.py     # DocumentLoader testleri
└── test_chunking_processor.py  # ChunkingProcessor testleri

examples/
└── chunking_example.py         # Kullanım örneği
```

## 🔍 Desteklenen Dosya Formatları

| Format | Uzantı | Açıklama |
|--------|--------|----------|
| PDF | `.pdf` | Portable Document Format |
| Word | `.docx` | Microsoft Word belgeleri |
| PowerPoint | `.pptx` | Microsoft PowerPoint sunumları |
| Excel | `.xlsx` | Microsoft Excel elektronik tabloları |

## ⚠️ Notlar

1. Büyük dosyalar için chunk_size ve chunk_overlap parametrelerini ihtiyacınıza göre ayarlayın
2. OCR gerektiren görsel PDF'ler için unstructured kütüphanesinin ek bağımlılıkları gerekebilir
3. PPTX ve XLSX dosyaları için unstructured kütüphanesi kullanılır
4. Chunk'lar arası örtüşme, bağlamsal bütünlüğü korumak için önemlidir

## 👥 Geliştirici

**Sorumlular**: Engin, Batuhan

## 📚 Daha Fazla Bilgi

- [Chunking Guide](../../docs/1_chunking_guide.md) - Detaylı teknik dokümantasyon
- [Complete Workflow](../../docs/5_complete_workflow.md) - Proje iş akışı

## 🐛 Sorun Bildirimi

Sorular ve öneriler için lütfen GitHub Issues bölümünü kullanın.
