"""
api_schemas.py — API istek/yanıt şemaları (kontrat dokümanı v0.1 ile uyumlu).

Bu şemalar, Teracity'ye iletilen API Kontrat Dokümanı'ndaki JSON yapılarının
Pydantic karşılığıdır. abac.py'deki iç modellerden (UserContext, AudiencePolicy)
ayrı tutulur; böylece dış kontrat ile iç mantık bağımsız evrilebilir.
"""

from __future__ import annotations
import base64
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

from abac import AudiencePolicy


# ── Uç Nokta 1: /query ────────────────────────────────────────────────────────
class QuerySource(BaseModel):
    dokuman_id: str
    dosya_adi: str
    sayfa: Optional[int] = None
    versiyon: Optional[int] = None   # (Faz 3 / madde 5) — doküman içerik sürümü
    skor: float


class QueryResponse(BaseModel):
    durum: str                       # "basarili" | "sonuc_yok"
    yanit: str
    kaynaklar: list[QuerySource] = []


class ConversationMessage(BaseModel):
    """Çok turlu konuşmada geçmiş bir tur (madde 13 kararı: history istemci
    tarafından her istekte gönderilir — sunucu taraflı oturum durumu yok)."""
    rol: Literal["kullanici", "asistan"]
    icerik: str = Field(..., min_length=1, max_length=8000)


class QueryRequest(BaseModel):
    soru: str = Field(..., min_length=1, max_length=2000, description="Kullanıcının sorusu")
    # Önceki turlar, KRONOLOJİK sırayla (en eski → en yeni). Boş/verilmezse
    # tek turluk davranış — geriye dönük uyumluluk korunur.
    gecmis_mesajlar: list[ConversationMessage] = Field(
        default_factory=list, max_length=20,
        description="Önceki konuşma turları (kronolojik sırayla, en fazla 20 mesaj)",
    )
    # Not: kullanici_baglami REQUEST GÖVDESİNDE DEĞİL, imzalı JWT içinden gelir
    # (bkz. auth.verify_user_context). Böylece istemci kendi yetkisini uyduramaz.


# ── Uç Nokta 2: /documents/{id}/audience ─────────────────────────────────────
class AudienceUpdateRequest(BaseModel):
    audience_policy: AudiencePolicy
    # (12) Boş politika = "kimse göremez". Kasıtlıysa true gönderilmelidir.
    allow_empty: bool = False
    # (Faz 4 / madde 11) Optimistic locking — mevcut audience_versiyon'la
    # eşleşmezse istek 409 ile reddedilir. Güncel değer GET /audience ile
    # öğrenilir.
    beklenen_audience_versiyon: int = Field(
        ..., description="Optimistic locking için mevcut hedef kitle sürümü"
    )
    # Denetim kaydı: değişikliği Bilimp arayüzünde yapan gerçek kullanıcının
    # kimliği. Servis token'ı bu isteği taşır ama işlemi bir insan tetikler;
    # bu alan olmadan "bu dokümanı kim, ne zaman, kime açtı" sorusu API
    # tarafında cevaplanamaz.
    degistiren_kullanici_id: int = Field(
        ..., description="Değişikliği Bilimp arayüzünde yapan kullanıcının kimliği"
    )


class AudienceUpdateResponse(BaseModel):
    durum: str                       # "basarili"
    dokuman_id: str
    guncellenen_kural_sayisi: int
    degistiren_kullanici_id: int
    yeni_audience_versiyon: int
    qdrant_sync: str                 # "tamamlandi"


class AudienceGetResponse(BaseModel):
    dokuman_id: str
    audience_policy: AudiencePolicy
    audience_versiyon: int   # (Faz 4 / madde 11) — sonraki PUT için beklenen_audience_versiyon burada


# ── Uyum İzleme: hedef kitlesi tanımlanmamış dokümanlar ──────────────────────
class ComplianceReportItem(BaseModel):
    """(Faz 3 / madde 8) — dokumanlar[] dizisinin eleman şeması netleştirildi."""
    dokuman_id: str
    versiyon: Optional[int] = None   # ilgili dokümanın en son bilinen içerik sürümü


class AudienceComplianceReportResponse(BaseModel):
    politikasiz_dokuman_sayisi: int   # TOPLAM sayı (tüm sayfalar dahil) — anlamı değişmedi
    limit: int                        # (Faz 4 / madde 16)
    offset: int                       # (Faz 4 / madde 16)
    dokumanlar: list[ComplianceReportItem]   # yalnızca BU SAYFAdaki elemanlar


# ── Uç Nokta: Doküman Oluşturma ──────────────────────────────────────────────
class DocumentCreateRequest(BaseModel):
    dokuman_id: str = Field(
        ..., min_length=1,
        description="Bilimp tarafından üretilen, tenant içinde benzersiz dosya adı (örn. 'rapor.pdf')",
    )
    dosya_icerigi_base64: str = Field(..., description="Dosyanın base64 kodlanmış ham baytları")
    audience_policy: AudiencePolicy
    yukleyen_kullanici_id: int = Field(
        ..., description="Yüklemeyi Bilimp arayüzünde yapan kullanıcının kimliği (denetim kaydı)"
    )

    @field_validator("dosya_icerigi_base64")
    @classmethod
    def _valid_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("dosya_icerigi_base64 geçerli bir base64 dizesi değil.")
        return v


class DocumentCreateResponse(BaseModel):
    durum: str                       # "basarili"
    dokuman_id: str
    versiyon: int
    chunk_sayisi: int


# ── Uç Nokta: İçerik Güncelleme (Sürümleme) ──────────────────────────────────
class DocumentContentUpdateRequest(BaseModel):
    dosya_icerigi_base64: str
    beklenen_versiyon: int = Field(..., description="Optimistic locking — mevcut sürümle eşleşmeli")
    degistiren_kullanici_id: int = Field(..., description="Değişikliği yapan kullanıcının kimliği")

    @field_validator("dosya_icerigi_base64")
    @classmethod
    def _valid_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("dosya_icerigi_base64 geçerli bir base64 dizesi değil.")
        return v


class DocumentContentUpdateResponse(BaseModel):
    durum: str                       # "basarili"
    dokuman_id: str
    yeni_versiyon: int
    chunk_sayisi: int


# ── Uç Nokta: Doküman Silme ───────────────────────────────────────────────────
class DocumentDeleteResponse(BaseModel):
    durum: str                       # "basarili"
    dokuman_id: str
    silinen_nokta_sayisi: int


# ── Uç Nokta: Toplu Hedef Kitle Güncelleme ────────────────────────────────────
class BulkAudienceUpdateItemRequest(BaseModel):
    """(Faz 4 / madde 12) Toplu güncellemedeki tek bir öğe."""
    dokuman_id: str
    audience_policy: AudiencePolicy
    beklenen_audience_versiyon: int
    allow_empty: bool = False


class BulkAudienceUpdateRequest(BaseModel):
    guncellemeler: list[BulkAudienceUpdateItemRequest] = Field(
        ..., min_length=1, max_length=100,
        description="Aynı istekte güncellenecek dokümanlar (en fazla 100)",
    )
    # Tüm öğeler için ORTAK denetim kaydı — toplu işlemi tetikleyen kullanıcı.
    degistiren_kullanici_id: int = Field(
        ..., description="Değişikliği Bilimp arayüzünde yapan kullanıcının kimliği"
    )


class BulkAudienceUpdateItemResult(BaseModel):
    dokuman_id: str
    durum: str                              # "basarili" | "hata"
    yeni_audience_versiyon: Optional[int] = None
    hata_kodu: Optional[str] = None         # "not_found" | "empty_rules" | "version_conflict"
    mesaj: Optional[str] = None


class BulkAudienceUpdateResponse(BaseModel):
    durum: str                       # "tamamlandi" (kısmi başarı da bu durumdadır)
    toplam: int
    basarili: int
    basarisiz: int
    sonuclar: list[BulkAudienceUpdateItemResult]