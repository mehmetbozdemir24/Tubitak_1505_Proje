# ABAC Tabanli Product-Ready Gelistirme Kilavuzu (On-Prem RAG)

## 1. Dokumanin Amaci
Bu dokuman, mevcut proje kod tabanini product-ready seviyeye cikarmak icin gelistirme ekibinin adim adim ne yapacagini tarif eder.

Hedef:
- Hiyerarsik rol mantigini kaldirmak
- Etiket/atribut eslesmesine dayali ABAC modeli kurmak
- Dokuman erisimini Qdrant tarafinda pre-filter ile garanti altina almak
- Teracity ile guvenli entegrasyon yapmak
- Sistemi test, izleme ve operasyon acisindan canli ortama hazirlamak

Bu rehber, kod degisikligi, test, migration, release ve rollback adimlarini birlikte kapsar.

---

## 2. Mevcut Durum Ozeti (As-Is)
Mevcut kodda ana noktalar:
- API sorgu modeli [api.py](../api.py) icinde role alanina dayaniyor.
- Filtreleme metadata.permission == role seklinde birebir yapiyor.
- UI [Tubitak_1505_UI/app.js](../Tubitak_1505_UI/app.js) tarafindan role client tarafinda seciliyor ve API'ye gonderiliyor.
- Upload akisinda yetki bilgisi role ile set ediliyor.

Bu yaklasim product-ready degildir cunku:
- Istemciden gelen role spoof edilebilir.
- Hiyerarsi kurallari ve esnek atribut kombinasyonlari desteklenmez.
- "Stajyer gorebilir ama yonetici goremez" gibi durumlar yalnizca ABAC ile dogru cozulur.

---

## 3. Hedef Durum (To-Be)

### 3.1 Yetki Modeli
Hiyerarsi yok, inheritance yok.

Sadece eslesme vardir:
- Dokuman tarafi: allow (opsiyonel deny)
- Kullanici tarafi: active_attributes
- Erişim: alan bazli kesisim kurallari ile belirlenir

### 3.2 Cekirdek Semantik
Alan ici OR, alanlar arasi AND:
- Ornek: department icinde ["pazarlama", "satis"] => kullanicida herhangi biri varsa o alan gecer
- Tum tanimli alanlar gecmek zorundadir

Bos alan kurali:
- Dokumanda bir alan bossa, o alan icin kisit yoktur (wildcard)

Opsiyonel exclusion (v2):
- deny alanlari tanimlanirsa, deny always wins kurali uygulanir

---

## 4. Resmi Erişim Kurali
Asagidaki tanimlar kullanilir:
- U_f: kullanicinin f alani icin atribut kumesi
- A_f: dokumanin allow[f] kumesi
- D_f: dokumanin deny[f] kumesi (opsiyonel)

Alan gecis kurali:
- pass_f = (A_f bos) OR (U_f ∩ A_f != bos)

Deny kurali:
- deny_f = (D_f dolu) AND (U_f ∩ D_f != bos)

Dokuman gorunurluk kurali:
- visible = (tum f alanlari icin pass_f true) AND (hicbir f alaninda deny_f true degil)

Uygulama notu:
- Qdrant aramasinda bu mantik filter objesi ile retrieval asamasinda uygulanir.
- Sonradan post-filter yapmak yalnizca ek savunma olabilir; ana kontrol DB seviyesinde olmalidir.

---

## 5. Standart Atribut Sozlugu
Tum entegrasyon taraflari ayni alan adlarini kullanmalidir:

- company
- branch
- department
- unit
- group
- building
- position
- employment_type
- custom_tags

Kural:
- Tum degerler normalize edilir: trim + lowercase
- Bos string degerler temizlenir
- Tekrarlanan degerler deduplicate edilir

---

## 6. API Sozlesmesi (V2)

### 6.1 Query Endpoint
Endpoint:
- POST /query

Request ornegi:

```json
{
  "question": "Pazarlama stajyer prim kurali nedir?",
  "user_id": "u-1024",
  "tenant_id": "teracity-main",
  "active_attributes": {
    "company": ["teracity"],
    "branch": ["ankara"],
    "department": ["pazarlama"],
    "employment_type": ["stajyer"],
    "position": ["intern"]
  },
  "doc_type": "pdf",
  "k": 5,
  "request_id": "req-2026-04-12-00001"
}
```

Response ornegi:

```json
{
  "answer": "...",
  "context_used": ["...", "..."],
  "processing_time_ms": 812.3,
  "answer_language": "tr",
  "question_language": "tr",
  "context_language": "tr",
  "language_source": "question",
  "trace": {
    "request_id": "req-2026-04-12-00001",
    "matched_points": 14,
    "returned_contexts": 3
  }
}
```

### 6.2 Audience Update Endpoint
Endpoint:
- PATCH /audience/update

Amaç:
- Yeniden embedding yapmadan mevcut point payload'indaki audience bilgisini guncellemek

Request ornegi:

```json
{
  "source": "Bilimp_Kullanim_Kilavuzu_Ornek_Sablon.txt",
  "audience": {
    "allow": {
      "department": ["pazarlama"],
      "employment_type": ["stajyer"]
    },
    "deny": {
      "position": ["director"]
    }
  },
  "updated_by": "admin-user-44"
}
```

### 6.3 Bulk Audience Update Endpoint
Endpoint:
- PATCH /audience/bulk-update

Amaç:
- Toplu dokuman hedef kitle guncellemesi

Not:
- Islem idempotent olmali
- Her kayit icin basarili/basarisiz durum donmeli

---

## 7. Teracity Entegrasyon Sozlesmesi
En guvenli iki model:

### Model A (Onerilen)
Teracity API sadece user_id gonderir, backend attributeleri server-to-server ceker.

Avantajlar:
- Client manipulasyonu azalir
- Tek dogru kaynak korunur

### Model B
Teracity active_attributes gonderir ama imzali token/JWT claim ile.

Avantajlar:
- Dusuk gecikme

Sart:
- Token imza dogrulama zorunlu
- Exp, iat, iss, aud kontrolleri zorunlu

Kesin kural:
- UI veya dis istemci tarafindan gelen role/attributes asla dogrulanmadan kullanilmaz.

---

## 8. Qdrant Payload Semasi
Her chunk payload'inda asagidaki alanlar olmali:

```json
{
  "metadata": {
    "source": "...",
    "file_type": "pdf",
    "language": "tr",
    "audience": {
      "allow": {
        "company": ["teracity"],
        "department": ["pazarlama"],
        "employment_type": ["stajyer"]
      },
      "deny": {
        "position": ["director"]
      },
      "version": 2,
      "updated_at": "2026-04-12T10:15:00Z"
    }
  }
}
```

V1 geri uyumluluk:
- metadata.permission alani bir sure tutulabilir
- Yeni ingestion surecinde audience zorunlu olmali

---

## 9. Dosya Bazli Gelistirme Adimlari

## 9.1 API Katmani
Hedef dosya:
- [api.py](../api.py)

Yapilacaklar:
1. QueryRequest modelini V2 sozlesmesine gore guncelle.
2. role alanini opsiyonel hale getir ve deprecated olarak isaretle.
3. active_attributes icin dogrulama fonksiyonu ekle.
4. ABAC filter builder fonksiyonu ekle.
5. similarity_search_with_score cagrisina filter parametresi ile pre-filter uygula.
6. /audience/update endpoint ekle.
7. /audience/bulk-update endpoint ekle.
8. request_id ureterek tum loglarda takip et.

Ornek fonksiyon iskeleti:

```python
def build_abac_filter(active_attributes: dict, doc_type: str | None = None):
    must_conditions = []
    must_not_conditions = []

    for field, user_values in active_attributes.items():
        if not user_values:
            continue

        must_conditions.append(
            models.FieldCondition(
                key=f"metadata.audience.allow.{field}",
                match=models.MatchAny(any=user_values)
            )
        )

        must_not_conditions.append(
            models.FieldCondition(
                key=f"metadata.audience.deny.{field}",
                match=models.MatchAny(any=user_values)
            )
        )

    if doc_type:
        must_conditions.append(
            models.FieldCondition(
                key="metadata.file_type",
                match=models.MatchValue(value=doc_type)
            )
        )

    return models.Filter(must=must_conditions, must_not=must_not_conditions)
```

Not:
- "Bos allow alan = kisit yok" kurali icin iki yaklasimdan birini secin:
  - Yaklasim 1: Her alani payload'da her zaman doldur (ANY gibi sentinel)
  - Yaklasim 2: Filter builder'da alan yoksa gec mantigi icin ek OR semantigi uygulayin

Oneri:
- V1'de sentinel kullanimi daha basit ve hataya daha kapali olur.

## 9.2 Ingestion ve Chunk Katmani
Hedef dosya:
- [chunker_module.py](../chunker_module.py)

Yapilacaklar:
1. metadata.permission yerine metadata.audience yapisini uret.
2. Varsayilan allow politikasini konfig dosyasindan al.
3. fixed_permission parametresini deprecated et, yerine audience parametresi ekle.
4. Her chunk'a audience.version ve updated_at yaz.
5. Eski veriler icin migration script yaz.

## 9.3 Dil Politikalari
Hedef dosya:
- [language_utils.py](../language_utils.py)

Yapilacaklar:
1. Mevcut soru/context tabanli dil secimi korunur.
2. Cevap dili seciminin audit kaydi request_id ile tutulur.
3. Hata mesaji ve no-answer metinleri TR/EN parity ile korunur.

## 9.4 Frontend Katmani
Hedef dosya:
- [Tubitak_1505_UI/app.js](../Tubitak_1505_UI/app.js)

Yapilacaklar:
1. Role secimi UI'dan kaldir (veya sadece demo modunda acik tut).
2. API isteklerine Authorization header ekle.
3. active_attributes verisini backend'den server-side almak oncelikli oldugu icin UI'dan tasima zorunlulugu olmasin.
4. innerHTML yerine textContent kullanarak XSS riskini azalt.

## 9.5 Altyapi ve Konfig
Hedef dosyalar:
- [docker-compose.yml](../docker-compose.yml)
- [requirements.txt](../requirements.txt)

Yapilacaklar:
1. API konteynerine ABAC/teracity env degiskenleri ekle.
2. JWT dogrulama kullanilacaksa gerekli paketleri ekle (ornek: python-jose).
3. CORS ayarini whitelist tabanli hale getir.
4. Dockerfile giris scriptini gercek dosya adi ile uyumlu hale getir.

---

## 10. Guvenlik Sertlestirme (Zorunlu)

## 10.1 Kimlik ve Token Dogrulama
Zorunlu kontroller:
- Signature verification
- exp, nbf, iat kontrolu
- iss ve aud kontrolu
- Clock skew toleransi

## 10.2 Upload Guvenligi
- Dosya adini sanitize et (path traversal engeli)
- Sadece izinli uzantilari kabul et
- Dosya boyutu limiti koy
- MIME ve uzanti tutarliligini denetle

## 10.3 API Guvenligi
- Rate limit
- Request size limiti
- Structured audit log
- Role/attribute spoof denemeleri icin guvenlik eventi

---

## 11. Test Plani (Eksiksiz)

## 11.1 Unit Testler
Test dosyasi onerisi:
- tests/test_abac_filter.py

Senaryolar:
1. Alan ici OR dogru mu
2. Alanlar arasi AND dogru mu
3. Bos allow alani wildcard gibi davraniyor mu
4. Deny varsa engelliyor mu
5. Normalization (trim/lowercase) dogru mu

## 11.2 Integration Testler
Test dosyasi onerisi:
- tests/test_query_authorization.py

Senaryolar:
1. Yetkili kullanici beklenen contexti aliyor mu
2. Yetkisiz kullanici sifir context aliyor mu
3. Ayni soruda farkli attribute setleri farkli sonuc veriyor mu
4. audience update sonrasi embedding olmadan erisim degisiyor mu

## 11.3 Security Testler
Senaryolar:
1. Sahte role gonderimi
2. Sahte active_attributes payload
3. Gecersiz token
4. Expired token
5. XSS payload denemesi

## 11.4 Performance Testler
Olcumler:
- p50, p95, p99 latency
- QPS
- Qdrant filter cost

Karsilastirma:
- Post-filter vs pre-filter
- Audience update once/sonra

---

## 12. Migration Plani

## 12.1 Asama 1: Dual-Read
- Hem metadata.permission hem metadata.audience okunur
- Feature flag: ABAC_ENABLED=false

## 12.2 Asama 2: Backfill
- Tum eski point payload'lari audience semasina tasinir
- Eksik alanlar varsayilan policy ile doldurulur

## 12.3 Asama 3: ABAC Aktivasyon
- ABAC_ENABLED=true
- permission tabanli filtreleme sadece fallback kalir

## 12.4 Asama 4: Temizlik
- permission alani kaldirilir
- Kod ve dokuman sadeleştirilir

---

## 13. Gozlemlenebilirlik ve Operasyon

## 13.1 Loglama
Her sorguda loglanacak alanlar:
- request_id
- user_id
- tenant_id
- filter_summary
- matched_points
- elapsed_ms
- model_name

## 13.2 Metrikler
- query_count_total
- query_latency_ms_bucket
- unauthorized_query_total
- audience_update_total
- audience_update_fail_total

## 13.3 Alarm Kurallari
- unauthorized_query_total anlik artis
- p95 latency threshold asimi
- qdrant hata oraninda artis

---

## 14. Sprint Bazli Uygulama Takvimi

## Sprint 1 (Guvenlik + Sozlesme)
- Query V2 modeli
- Token/attribute dogrulama
- ABAC filter builder
- Unit testlerin ilk seti

Cikis kriteri:
- role tabanli temel akistan ABAC akisa gecis

## Sprint 2 (Audience Yonetimi + Migration)
- /audience/update
- /audience/bulk-update
- payload backfill script
- integration testler

Cikis kriteri:
- Embedding olmadan yetki guncelleme canli testte dogrulandi

## Sprint 3 (Hardening + Release)
- UI sertlestirme
- CORS whitelist
- monitoring/alerts
- load test

Cikis kriteri:
- production readiness checklist tamam

---

## 15. Definition of Done (DoD)
Tum maddeler true olmadan release yok:

1. ABAC kurallari unit testlerde %100 kritik senaryo kapsami ile geciyor.
2. Yetkisiz kullaniciya retrieval seviyesinde veri donmuyor.
3. audience update embedding olmadan etkili.
4. Token dogrulama aktif ve bypass edilemiyor.
5. UI tarafinda role spoof yolu kapali.
6. p95 latency hedefi saglaniyor.
7. Rollback plani dry-run edildi.

---

## 16. Rollback Plani
Probleme girildiginde:
1. Feature flag ile ABAC'i kapat, V1 filtreye don.
2. Son deployment'i geri al.
3. Incident kaydi ac.
4. Etkilenen request_id araligini auditten cikar.
5. Koku neden analizi tamamlanmadan tekrar acma.

---

## 17. Ekip Icine Kisa Gorev Dagilimi Onerisi
- API/ABAC: 1 backend geliştirici
- Ingestion/Migration: 1 backend-data geliştirici
- Frontend sertlestirme: 1 frontend geliştirici
- Test/CI: 1 QA veya SDET
- Operasyon/izleme: 1 DevOps

---

## 18. Hemen Baslanacak Is Listesi (Ilk 72 Saat)
1. Query V2 modelini [api.py](../api.py) icinde acin.
2. ABAC filter builder fonksiyonunu ekleyin.
3. role bazli filtreyi feature flag altina alin.
4. audience payload semasini [chunker_module.py](../chunker_module.py) icine ekleyin.
5. /audience/update endpoint'ini aktif edin.
6. Unit testleri yazin ve CI'a baglayin.
7. Demo UI'da role secimini kapatin.
8. CORS ve upload guvenligini sertlestirin.

Bu 8 adim tamamlandiginda sistem product-ready yolunda kritik esigi gecmis olur.
