from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Table
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship

# ---  VERİTABANI BAĞLANTISI ---
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:12345@localhost:5432/testDB"  #sistemin mevcut dbsiyle değişmeli.
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- DİNAMİK YETKİ KÖPRÜ TABLOLARI ---
kullanici_ekler_tablosu = Table(
    "kullanici_ekler", Base.metadata,
    Column("kullanici_id", Integer, ForeignKey("kullanicilar.kullanici_id")),
    Column("ek_id", Integer, ForeignKey("ekler.ek_id"))
)

mudurluk_ekler_tablosu = Table(
    "mudurluk_ekler", Base.metadata,
    Column("mudurluk_id", Integer, ForeignKey("mudurlukler.mudurluk_id")),
    Column("ek_id", Integer, ForeignKey("ekler.ek_id"))
)

pozisyon_ekler_tablosu = Table(
    "pozisyon_ekler", Base.metadata,
    Column("pozisyon_id", Integer, ForeignKey("pozisyonlar.pozisyon_id")),
    Column("ek_id", Integer, ForeignKey("ekler.ek_id"))
)

''' kullanici_ekler_tablosu: Hangi kullanıcının hangi dokümana doğrudan erişimi olduğunu tutar.

mudurluk_ekler_tablosu: Hangi departmanın (müdürlüğün) hangi dokümanları görebileceğini tutar.

pozisyon_ekler_tablosu: Hangi pozisyonun (örneğin Genel Müdür) hangi dokümanlara erişebileceğini tutar. Bu yapı ihtiyaca göre genişletilebilir. '''

# --- MODELLER ---
class Mudurluk(Base):
    __tablename__ = "mudurlukler"
    mudurluk_id = Column(Integer, primary_key=True)
    mudurluk_adi = Column(String)
    ekler = relationship("Ek", secondary=mudurluk_ekler_tablosu) # Departmanın erişebildiği dokümanlar

class Pozisyon(Base):
    __tablename__ = "pozisyonlar"
    pozisyon_id = Column(Integer, primary_key=True)
    pozisyon_adi = Column(String)
    ekler = relationship("Ek", secondary=pozisyon_ekler_tablosu) # Pozisyonun erişebildiği dokümanlar

class Ek(Base):
    __tablename__ = "ekler"
    ek_id = Column(Integer, primary_key=True)
    ek_adi = Column(String)

class Kullanici(Base):
    __tablename__ = "kullanicilar"
    kullanici_id = Column(Integer, primary_key=True)
    kullanici_adi = Column(String)
    mudurluk_id = Column(Integer, ForeignKey("mudurlukler.mudurluk_id"))
    pozisyon_id = Column(Integer, ForeignKey("pozisyonlar.pozisyon_id"))
    
    mudurluk = relationship("Mudurluk")
    pozisyon = relationship("Pozisyon")
    ekler = relationship("Ek", secondary=kullanici_ekler_tablosu) # Doğrudan zimmetli dokümanlar


app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class AgentYetkiIstegi(BaseModel):
    kullanici_adi: str
    dokuman_adi: str

class AgentYetkiCevabi(BaseModel):
    yetki_var_mi: bool
    mesaj: str

# --- LLM UÇ NOKTASI ---
@app.post("/api/v1/agent/yetki-kontrol", response_model=AgentYetkiCevabi, tags=["API"])
def ai_ajan_yetki_kontrolu(istek: AgentYetkiIstegi, db: Session = Depends(get_db)):
   
    kullanici = db.query(Kullanici).filter(Kullanici.kullanici_adi == istek.kullanici_adi).first()
    if not kullanici:
        return AgentYetkiCevabi(yetki_var_mi=False, mesaj="Kullanıcı bulunamadı.")

    # Doküman adının içinde geçip geçmediğine bakar (Örn: "abc" yazarsa "abc.pdf"i bulur)
    dokuman = db.query(Ek).filter(Ek.ek_adi.ilike(f"%{istek.dokuman_adi}%")).first()
    if not dokuman:
        return AgentYetkiCevabi(yetki_var_mi=False, mesaj="Doküman bulunamadı.")

    # 1. KONTROL: Doğrudan Zimmetli mi?
    if dokuman in kullanici.ekler:
        return AgentYetkiCevabi(yetki_var_mi=True, mesaj="Doğrudan erişim yetkisi onaylandı.")

    # 2. KONTROL: Departman Yetkisi var mı?
    if kullanici.mudurluk and dokuman in kullanici.mudurluk.ekler:
        return AgentYetkiCevabi(yetki_var_mi=True, mesaj=f"Departman yetkisi ile erişim sağlandı ({kullanici.mudurluk.mudurluk_adi}).")

    # 3. KONTROL: Pozisyon Yetkisi var mı?
    if kullanici.pozisyon and dokuman in kullanici.pozisyon.ekler:
        return AgentYetkiCevabi(yetki_var_mi=True, mesaj=f"Pozisyon yetkisi ile erişim sağlandı ({kullanici.pozisyon.pozisyon_adi}).")

    # Hiçbir yetki yoksa Reddet!
    return AgentYetkiCevabi(yetki_var_mi=False, mesaj="Gizlilik kuralı: Bu dokümana erişim yetkiniz bulunmamaktadır.")