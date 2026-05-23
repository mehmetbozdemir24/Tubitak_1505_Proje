Veritabanı (PostgreSQL) Kurulumu
postgrede testDB adında boş bir veritabanı oluşturun.

Query Tool'u açarak aşağıdaki SQL kodunu çalıştırın. Bu işlem tabloları oluşturacak ve test verilerini basacaktır.

------------------------------------------

DROP SCHEMA public CASCADE;
CREATE SCHEMA public;

-- 1.  Gerekli  Referans Tabloları
CREATE TABLE sirketler (sirket_id INT PRIMARY KEY, sirket_adi VARCHAR);
CREATE TABLE subeler (sube_id INT PRIMARY KEY, sube_adi VARCHAR);
CREATE TABLE mudurlukler (mudurluk_id INT PRIMARY KEY, mudurluk_adi VARCHAR);
CREATE TABLE birimler (birim_id INT PRIMARY KEY, birim_adi VARCHAR);
CREATE TABLE gruplar (grup_id INT PRIMARY KEY, grup_adi VARCHAR);
CREATE TABLE binalar (bina_id INT PRIMARY KEY, bina_adi VARCHAR);
CREATE TABLE pozisyonlar (pozisyon_id INT PRIMARY KEY, pozisyon_adi VARCHAR);
CREATE TABLE personel_tipleri (personel_tip_id INT PRIMARY KEY, personel_tip_adi VARCHAR);
CREATE TABLE ekler (ek_id SERIAL PRIMARY KEY, ek_adi VARCHAR);

-- 2. Ana Kullanıcı Tablosu
CREATE TABLE kullanicilar (
    kullanici_id INT PRIMARY KEY,
    kullanici_adi VARCHAR,
    sirket_id INT REFERENCES sirketler(sirket_id),
    sube_id INT REFERENCES subeler(sube_id),
    mudurluk_id INT REFERENCES mudurlukler(mudurluk_id),
    birim_id INT REFERENCES birimler(birim_id),
    grup_id INT REFERENCES gruplar(grup_id),
    bina_id INT REFERENCES binalar(bina_id),
    pozisyon_id INT REFERENCES pozisyonlar(pozisyon_id),
    personel_tip_id INT REFERENCES personel_tipleri(personel_tip_id)
);

-- 3. Dinamik Yetki Köprü Tabloları (RBAC İçin)
CREATE TABLE kullanici_ekler (kullanici_id INT REFERENCES kullanicilar(kullanici_id), ek_id INT REFERENCES ekler(ek_id), PRIMARY KEY (kullanici_id, ek_id));
CREATE TABLE mudurluk_ekler (mudurluk_id INT REFERENCES mudurlukler(mudurluk_id), ek_id INT REFERENCES ekler(ek_id), PRIMARY KEY (mudurluk_id, ek_id));
CREATE TABLE pozisyon_ekler (pozisyon_id INT REFERENCES pozisyonlar(pozisyon_id), ek_id INT REFERENCES ekler(ek_id), PRIMARY KEY (pozisyon_id, ek_id));

-- 4. API Test Verileri
INSERT INTO mudurlukler VALUES (10, 'Analiz ve Destek'), (13, 'Yazılım'), (16, 'İnsan Kaynakları'), (17, 'Finans');
INSERT INTO pozisyonlar VALUES (1, 'Genel Müdür'), (108, 'Tasarım Yöneticisi');
INSERT INTO ekler VALUES (1, 'abc.pdf'), (17, 'Q1_Finans_Raporu.pdf'), (18, 'IK_Politikalari.pdf');

INSERT INTO kullanicilar (kullanici_id, kullanici_adi, mudurluk_id, pozisyon_id) VALUES 
(590, 'c.sucu', 10, 108), (800, 'm.yilmaz', 13, NULL), (801, 'a.demir', 16, NULL), (802, 's.celik', 17, 1);

INSERT INTO kullanici_ekler VALUES (590, 1);
INSERT INTO mudurluk_ekler VALUES (17, 17), (16, 18);
INSERT INTO pozisyon_ekler VALUES (1, 1);

-- 5. İŞTE O 19 SÜTUNLUK GÖRÜNÜM (VIEW)
CREATE VIEW orjinal_veri_gorunumu AS
SELECT 
    e.ek_adi AS "Ek Adı", s.sirket_id AS "Sirket ID", s.sirket_adi AS "Sirket Adi",
    sb.sube_id AS "SubeID", sb.sube_adi AS "SubeAdi", m.mudurluk_id AS "MudurlukID",
    m.mudurluk_adi AS "MudurlukAdi", b.birim_id AS "BirimID", b.birim_adi AS "BirimAdi",
    g.grup_id AS "Grup ID", g.grup_adi AS "GrupAdi", bn.bina_id AS "BinaID",
    bn.bina_adi AS "BinaAdi", p.pozisyon_id AS "PozisyonID", p.pozisyon_adi AS "PozisyonAdi",
    pt.personel_tip_id AS "Personel TipID", pt.personel_tip_adi AS "PersonelTipAdi",
    k.kullanici_id AS "KullanıcıID", k.kullanici_adi AS "KullanıcıAdi"
FROM kullanicilar k
JOIN kullanici_ekler ke ON k.kullanici_id = ke.kullanici_id
JOIN ekler e ON ke.ek_id = e.ek_id
LEFT JOIN sirketler s ON k.sirket_id = s.sirket_id
LEFT JOIN subeler sb ON k.sube_id = sb.sube_id
LEFT JOIN mudurlukler m ON k.mudurluk_id = m.mudurluk_id
LEFT JOIN birimler b ON k.birim_id = b.birim_id
LEFT JOIN gruplar g ON k.grup_id = g.grup_id
LEFT JOIN binalar bn ON k.bina_id = bn.bina_id
LEFT JOIN pozisyonlar p ON k.pozisyon_id = p.pozisyon_id
LEFT JOIN personel_tipleri pt ON k.personel_tip_id = pt.personel_tip_id;

--------------------------------------------------

API Kurulumu ve Çalıştırma
Veritabanı hazırlandıktan sonra FastAPIyi başlatabilirsiniz:

Gerekli kütüphaneler:

--------

pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic

--------

API'yi ayağa kaldırın:

uvicorn api:app --reload

-------------

Sistem çalıştıktan sonra [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) adresine giderek POST /api/v1/agent/yetki-kontrol uç noktasını test edebilirsiniz.

Senaryo 1 (Yetkili): c.sucu, kendi zimmetindeki abc.pdf dosyasını ister. (Sonuç: Doğrudan Erişim)

Senaryo 2 (Departman Yetkisi): a.demir (İK personeli), IK_Politikalari.pdf dosyasını ister. (Sonuç: Erişim Onaylandı)

Senaryo 3 (Yetkisiz Erişim/Bloklama): m.yilmaz (Yazılımcı), gizli olan Q1_Finans_Raporu.pdf dosyasını okumaya çalışır. (Sonuç: Erişim Reddedildi)

-----------------

Test İçin json

Senaryo 1: Doğrudan Zimmetli Dosya Testi
{
  "kullanici_adi": "c.sucu",
  "dokuman_adi": "abc.pdf"
}

Senaryo 2: Departman Yetkisi Testi
{
  "kullanici_adi": "a.demir",
  "dokuman_adi": "IK_Politikalari"
}

Senaryo 3: Pozisyon (Rol) Yetkisi Testi
{
  "kullanici_adi": "s.celik",
  "dokuman_adi": "Müşteri Listesi"
}

--------------------------