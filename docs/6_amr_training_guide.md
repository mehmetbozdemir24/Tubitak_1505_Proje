# 🧩 AMR Model Eğitimi Rehberi

Bu doküman, mevcut TÜBİTAK 1505 RAG projesinin bağlamında **AMR (Abstract Meaning Representation — Soyut Anlam Temsili)** üreten bir modelin nasıl eğitileceğini, mevcut projede hangi modellerin kullanıldığını ve AMR için uygun veri setlerini açıklamaktadır.

---

## 📌 Mevcut Projede Kullanılan Modeller ve Veri Seti

### Embedding Modeli
| Özellik | Detay |
|---|---|
| **Model Adı** | `ytu-ce-cosmos/turkish-e5-large` |
| **Tür** | Bi-Encoder (Sentence Embedding) |
| **Vektör Boyutu** | 1024 |
| **Dil Desteği** | Türkçe + İngilizce |
| **Kullanım Amacı** | Metin parçalarını vektöre dönüştürme (Semantic Search) |

### Büyük Dil Modeli (LLM)
| Özellik | Detay |
|---|---|
| **Birincil Model** | `gemma3:12b` (Ollama üzerinden çalışır) |
| **İkincil Model** | `qwen3:8b` |
| **Alternatif** | `gemini-2.5-flash` (Google API) |
| **Kullanım Amacı** | Bağlam + soru → Doğal dil yanıtı üretimi |

### Proje Veri Seti
Mevcut proje, kurumsal belgelerden oluşan özel bir veri seti kullanmaktadır:

| Dosya Türü | Adet | Toplam Chunk |
|---|---|---|
| PDF | 33 dosya | ~530 chunk |
| Excel (.xlsx) | 3 dosya | 3 chunk (tablo) |
| Word (.docx) | 4 dosya | ~97 chunk |
| PowerPoint | 1 dosya | 3 chunk |
| **Toplam** | **41 dosya** | **~640 chunk** |

---

## 🔍 AMR (Abstract Meaning Representation) Nedir?

AMR, cümlenin anlamını yönlü bir graf yapısıyla temsil eden bir formalizmdur. Kelime sırasından bağımsız olarak anlam içeriğini kodlar.

**Örnek:**
```
Girdi: "The boy wants to go to the store."

AMR Grafı:
(z0 / want-01
   :ARG0 (z1 / boy)
   :ARG1 (z2 / go-02
      :ARG0 z1
      :destination (z3 / store)))
```

---

## 🤖 AMR Üretimi İçin Önerilen Modeller

### 1. `facebook/bart-large` — SPRING Yaklaşımı ⭐ (Önerilen)
- AMR parsing için en yaygın ve başarılı model
- SPRING makalesi: *"One SPRING to Rule Them All"* (Bevilacqua et al., 2021)
- **Avantaj**: Yüksek Smatch skoru (~84–86 F1 AMR 3.0'da)
- **Dezavantaj**: Yalnızca İngilizce eğitilmiş

```python
model_name = "facebook/bart-large"
```

### 2. `google/mt5-base` — Çok Dilli (Türkçe Dahil) ⭐
- mT5 (multilingual T5) 101 dil destekler, Türkçe dahil
- Türkçe AMR için cross-lingual transfer öğrenmesi uygulanabilir
- **Avantaj**: Türkçe metin için daha uygun
- **Dezavantaj**: BART'a göre daha düşük AMR performansı

```python
model_name = "google/mt5-base"  # veya "google/mt5-large"
```

### 3. `t5-base` / `t5-large` — T5 Modeli
- Seq2seq AMR parsing için sağlam temel
- AMR 3.0 üzerinde fine-tune edilmiş versiyonlar Hugging Face Hub'da mevcut

```python
model_name = "t5-base"
```

---

## 📊 AMR Eğitimi İçin Kullanılabilecek Veri Setleri

### Lisanslı (LDC) Veri Setleri — En Kapsamlı

| Veri Seti | Cümle Sayısı | Dil | Notlar |
|---|---|---|---|
| **LDC2020T02 (AMR 3.0)** | ~59.000 | İngilizce | En güncel, önerilen |
| **LDC2017T10 (AMR 2.0)** | ~36.521 | İngilizce | Yaygın kullanılan önceki sürüm |

> 📋 LDC verilerine erişim için: https://catalog.ldc.upenn.edu/

### Açık Erişim Veri Setleri — Ücretsiz

| Veri Seti | Cümle Sayısı | Dil | Erişim |
|---|---|---|---|
| **Little Prince AMR** | 1.562 | İngilizce | GitHub (açık) |
| **Bio AMR Corpus** | ~10.000 | İngilizce (biyomedikal) | LDC (ücretsiz) |
| **The Proxy Report** | ~8.000 | İngilizce | GitHub (açık) |

### Türkçe AMR için Yaklaşımlar

| Yaklaşım | Açıklama |
|---|---|
| **BOUN-TUPA** | Boğaziçi Üni. Türkçe Bağımlılık Parseri — AMR benzeri yapı |
| **Cross-lingual Transfer** | İngilizce AMR üzerinde BART/mT5 eğit → Türkçe'ye transfer |
| **Machine Translation** | Türkçe → İngilizce çevir → AMR parse → geri map et |
| **Paralel Veri Oluşturma** | İngilizce AMR + çeviri modeli ile sentetik Türkçe veri |

---

## 🚀 Hızlı Başlangıç

### 1. Gerekli Kütüphaneleri Kur

```bash
pip install transformers datasets torch accelerate penman smatch
```

### 2. Veriyi Hazırla (JSONL Formatı)

Her satır şu formatda olmalıdır:
```json
{"sentence": "The boy wants to go.", "amr": "(z0 / want-01 :ARG0 (z1 / boy) :ARG1 (z2 / go-02 :ARG0 z1))"}
```

### 3. Eğitimi Başlat

```bash
python amr_training.py
```

Veya özelleştirilmiş ayarlarla:

```python
from amr_training import AMRTrainingConfig, train

config = AMRTrainingConfig(
    model_name="google/mt5-base",     # Türkçe için
    train_file="data/amr/train.jsonl",
    dev_file="data/amr/dev.jsonl",
    num_train_epochs=30,
    per_device_train_batch_size=4,
    fp16=True,                         # GPU varsa True
    output_dir="amr_model_turkce"
)
train(config)
```

### 4. Çıkarsama

```python
from amr_training import predict_amr

amr = predict_amr(
    sentence="The boy wants to go to school.",
    model_dir="amr_model_output"
)
print(amr)
```

---

## 📐 Değerlendirme Metrikleri

AMR parsing kalitesi **Smatch skoru** ile ölçülür:

```bash
pip install smatch

# Smatch hesaplama
python -m smatch -f predicted.amr gold.amr --pr
```

| Model | Veri Seti | Smatch F1 |
|---|---|---|
| BART-large (SPRING) | AMR 3.0 | ~84–86 |
| T5-large | AMR 3.0 | ~81–83 |
| mT5-base | AMR 3.0 (cross-lingual) | ~70–75 |

---

## 🏗️ Mevcut Proje ile Entegrasyon

AMR modeli, mevcut RAG sistemine şu şekilde entegre edilebilir:

```
Kullanıcı Sorusu
     ↓
[AMR Parser] → Anlamsal Graf
     ↓
[AMR → Sorgu Dönüşümü]
     ↓
[Qdrant Vektör Araması] (ytu-ce-cosmos/turkish-e5-large)
     ↓
[LLM Yanıt Üretimi] (gemma3:12b / qwen3:8b)
     ↓
Kullanıcıya Yanıt
```

Bu yaklaşımla sorgunun anlam yapısı çıkarılarak daha hassas retrieval yapılabilir.

---

## 📚 Faydalı Kaynaklar

- [SPRING: AMR Parsing](https://github.com/SapienzaNLP/spring)
- [AMR Annotation Guidelines](https://amr.isi.edu/language.html)
- [Hugging Face AMR Modelleri](https://huggingface.co/models?search=amr)
- [LDC AMR Corpus](https://catalog.ldc.upenn.edu/LDC2020T02)
- [Smatch Evaluation](https://github.com/snowblink14/smatch)
