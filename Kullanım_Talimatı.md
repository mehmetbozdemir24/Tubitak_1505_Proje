Haklısın, sen direkt **README.md içeriği** olarak yapıştırmak istiyorsun; ben de üstüne bir de ```markdown bloğu koyunca karıştı. Aşağıdaki metni **tamamını** kopyalayıp `README.md` dosyana yapıştırabilirsin:

---

# 🚀 TÜBİTAK 1505 - RAG ve Embedding Modülü

Bu proje, **TÜBİTAK 1505** projesi kapsamında geliştirilmiş olup; yapılandırılmamış ve yarı yapılandırılmış verilerin (PDF, Excel, Word vb.) işlenmesi, vektör veritabanına kaydedilmesi ve **Büyük Dil Modelleri (LLM)** ile sorgulanmasını sağlayan **RAG (Retrieval-Augmented Generation)** mimarisini içerir.

Proje, dokümanları parçalar (chunking), anlamlandırır (embedding) ve kullanıcının sorularına yapay zeka destekli cevaplar üretir.

---

## 📋 Proje Özellikleri

* **Veri Kaynağı:** `Tubitak_1505_Proje/` klasörü altındaki önceden işlenmiş `.pkl` formatındaki verileri okur.
* **Embedding (Vektörleştirme):** Türkçe için optimize edilmiş **`ytu-ce-cosmos/turkish-e5-large`** modelini kullanır.
* **Vektör Veritabanı:** **Qdrant** veritabanını Docker üzerinde ayağa kaldırır ve yüksek performanslı vektör araması yapar.
* **Akıllı Sorgulama:** Kullanıcı sorularını yerel LLM **Google Gemma-3 (12B)** modeli ile cevaplar.
* **Hibrit Filtreleme:** Aramaları dosya türüne göre (örneğin *“Sadece Excel tablolarında ara”*) filtreleyebilir.

---

## 🛠️ Ön Gereksinimler (Prerequisites)

Projeyi çalıştırmadan önce bilgisayarınızda aşağıdaki araçların kurulu olması gerekmektedir:

1. **Docker Desktop:** Qdrant veritabanını konteynerize olarak çalıştırmak için gereklidir.
   [İndir](https://www.docker.com/products/docker-desktop/)
2. **Python 3.10 veya üzeri:** Kodları çalıştırmak için.
3. **Ollama:** LLM modelini (Gemma) yerelde çalıştırmak için.
   [İndir](https://ollama.com/)
4. **Git:** Projeyi klonlamak için.

---

## 🤖 LLM Modelinin Kurulumu

Bu proje `gemma3:12b` modelini kullanmaktadır. Terminalinizde (CMD veya PowerShell) şu komutu çalıştırarak modeli indirin:

```bash
ollama pull gemma3:12b
```

---

## ⚙️ Kurulum (Installation)

### 1. Projeyi Klonlayın

Terminali açın ve projeyi bilgisayarınıza indirin (eğer `embedding` dalındaysanız o dala geçiş yapın):

```bash
git clone https://github.com/KULLANICI_ADIN/REPO_ADIN.git
cd REPO_ADIN
git checkout embedding
```

### 2. Sanal Ortam Oluşturun (Önerilen)

Proje bağımlılıklarını izole etmek için sanal ortam kurun:

```bash
# Windows için:
python -m venv .venv
.\.venv\Scripts\activate

# Mac/Linux için:
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Kütüphaneleri Yükleyin

Gerekli Python paketlerini yükleyin:

```bash
pip install -r requirements.txt
```

---

## 🚀 Çalıştırma (How to Run)

Sistemi çalıştırmak için aşağıdaki iki adımı izleyin.

### Adım 1: Qdrant Veritabanını Başlatın (Docker)

Proje ana dizinindeyken (yani `docker-compose.yml` dosyasının olduğu yerde) şu komutu çalıştırın. Bu komut, Qdrant servisini başlatacaktır:

```bash
docker-compose up -d --build
```

**Kontrol:** Tarayıcınızdan `http://localhost:6333/dashboard` adresine giderek Qdrant arayüzünün çalıştığını görebilirsiniz.

### Adım 2: RAG Sistemini Çalıştırın

Veritabanı hazır olduktan sonra, verileri yüklemek ve soru sormak için ana Python dosyasını çalıştırın:

```bash
python TEST.py
```

> Not: Dosya adı projenizde `rag_test.py` ise komutu ona göre düzenleyiniz.

---

## 📂 Proje Klasör Yapısı

```text
PROJE_ANA_DIZIN/
│
├── Tubitak_1505_Proje/                  # Veri Klasörü
│   └── tum_dokumanlar_final_last.pkl    # Kaynak Chunk Verileri
│
├── TEST.py                              # Ana Çalıştırma Dosyası (Embedding + Chat)
├── docker-compose.yml                   # Docker konfigürasyonu (Qdrant Servisi)
├── Dockerfile                           # Python ortamı için Docker imaj tarifi
├── requirements.txt                     # Python kütüphane listesi
├── .gitignore                           # Git tarafından yoksayılacak dosyalar
└── README.md                            # Proje dokümantasyonu
```

---

## 🔍 Kullanım ve Filtreleme

Script çalıştırıldığında, kod içerisindeki `if __name__ == "__main__":` bloğundaki ayara göre hareket eder.

**Örnek Kod Kullanımı (`TEST.py` içinde):**

```python
if __name__ == "__main__":
    # Örnek 1: Genel Arama (Tüm dokümanlarda)
    soru = "Projenin genel bütçesi nedir?"
    run_rag_pipeline(soru)
    
    # Örnek 2: Sadece Excel Dosyalarında Arama (Filtreli)
    soru = "Ocak ayı satış rakamları nedir?"
    run_rag_pipeline(soru, doc_type="excel")
```

---

## ⚠️ Sık Karşılaşılan Hatalar ve Çözümleri

| Hata Mesajı                 | Olası Sebep                              | Çözüm                                                                 |
| --------------------------- | ---------------------------------------- | --------------------------------------------------------------------- |
| `Qdrant connection refused` | Docker konteyneri çalışmıyor.            | `docker-compose up -d` komutunu çalıştırın.                           |
| `CUDA not available`        | NVIDIA GPU bulunamadı veya sürücü eksik. | Sorun değil, sistem otomatik olarak CPU modunda çalışmaya devam eder. |
| `Ollama connection error`   | Ollama uygulaması kapalı.                | Ollama uygulamasını başlatın ve arka planda çalıştığından emin olun.  |
| `ModuleNotFoundError`       | Kütüphaneler eksik.                      | `pip install -r requirements.txt` komutunu tekrar çalıştırın.         |

---

## 👥 Katkıda Bulunanlar

* TÜBİTAK 1505 Proje Ekibi
