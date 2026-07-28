"""
tenancy.py — Çoklu Hesap (Multi-Tenant) İzolasyon Katmanı

TASARIM KARARI (v2.0) — Müşteri Seviyesinde Fiziksel İzolasyon
──────────────────────────────────────────────────────────────────────────────
İLK sürümde (Seviye C) fiziksel izolasyon sirket_id (şirket) bazında
kuruluydu. Teracity'nin bulgusu bunu geçersiz kıldı: Bilimp'in HER MÜŞTERİSİ
kendi ayrı veritabanına sahip; şirket/kullanıcı gibi kimlikler yalnızca o
müşterinin kendi veritabanı içinde üretiliyor ve GLOBAL OLARAK BENZERSİZ
DEĞİL. İki farklı Bilimp müşterisinde aynı sirket_id (hatta aynı
kullanici_id) rastlantısal olarak çakışabilir — sirket_id'yi fiziksel sınır
olarak kullanmak, iki farklı müşterinin verisinin AYNI koleksiyonda
karışmasına yol açardı.

Doğru izolasyon sınırı: musteri_id — Bilimp'in ürettiği, global benzersiz
müşteri (hesap) kimliği. Bir sorgu, kendi müşterisinin koleksiyonu DIŞINDA
hiçbir veriye erişemez; bu bir kural ihlaliyle değil, bir Qdrant koleksiyon
adıyla garanti edilir.

sirket_id ise artık şube/bina gibi SIRADAN bir hedef kitle özniteliğidir
(bkz. abac.py) ve normal şekilde ABAC filtresine katılır — koleksiyon içinde
HİÇ hariç tutulmaz, çünkü artık gereksiz/yedek bir alan değil, gerçek bir
erişim kısıtlamasıdır (bir müşterinin birden fazla şirketi olabilir).

──────────────────────────────────────────────────────────────────────────────
SOLID Notları

  SRP  — İsimlendirme (TenantRegistry), fiziksel provizyon
         (TenantCollectionProvisioner) ve çözümleme (resolve_tenant_collection)
         ayrı sorumluluklar olarak ayrıştırıldı.
  OCP  — Yeni bir müşteri eklemek hiçbir kod değişikliği gerektirmez;
         ConventionTenantRegistry kimlikten koleksiyon adını türetir.
  LSP  — TenantRegistry bir Protocol'dür; herhangi bir uyumlu implementasyon
         (örn. ileride bir veritabanı destekli kayıt) yerine geçebilir.
  ISP  — TenantRegistry yalnızca isimlendirme sorumluluğu taşır; provizyon
         (koleksiyon oluşturma) ayrı bir arayüzde (TenantCollectionProvisioner).
  DIP  — Alt seviye sınıflar (TenantRegistry, TenantCollectionProvisioner)
         BİLEREK "tenant_id" gibi genel bir isim kullanır — tenant kavramının
         NEYİ temsil ettiğini (şirket mi, müşteri mi) bilmezler; bu, tenant
         kavramı bir kez değiştiğinde (v1→v2) yaşanan acıyı tekrar
         yaşamamak içindir. Somut iş anlamı (musteri_id) yalnızca üst
         seviye giriş noktasında (resolve_tenant_collection) ve çağıran
         katmanlarda (api.py, rag_service.py) görünür.
"""

from __future__ import annotations
import re
from typing import Optional, Protocol, runtime_checkable

from fastapi import HTTPException, status
from qdrant_client import QdrantClient


# ══════════════════════════════════════════════════════════════════════════════
# İsimlendirme
# ══════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class TenantRegistry(Protocol):
    """Bir tenant kimliğini, o tenant'a ait Qdrant koleksiyon adına çözümler."""

    def collection_name(self, tenant_id: int) -> str: ...


class ConventionTenantRegistry:
    """
    Sabit bir isimlendirme kuralına göre koleksiyon adı üretir
    ("tubitak1505_musteri_{id}"). Yeni bir müşteri eklemek için kod
    değişikliği veya kayıt işlemi GEREKMEZ — kimlikten otomatik türetilir (OCP).
    """

    _PREFIX = "tubitak1505_musteri_"
    _VALID_SUFFIX = re.compile(r"^\d+$")

    def collection_name(self, tenant_id: int) -> str:
        if tenant_id is None or tenant_id < 0:
            raise ValueError(f"Geçersiz tenant kimliği: {tenant_id!r}")
        return f"{self._PREFIX}{tenant_id}"

    def tenant_id_from_collection(self, collection_name: str) -> Optional[int]:
        """Ters çözümleme — migrasyon ve tanılama araçları için."""
        if not collection_name.startswith(self._PREFIX):
            return None
        suffix = collection_name[len(self._PREFIX):]
        return int(suffix) if self._VALID_SUFFIX.match(suffix) else None


# ══════════════════════════════════════════════════════════════════════════════
# Fiziksel Provizyon
# ══════════════════════════════════════════════════════════════════════════════
class TenantCollectionProvisioner:
    """
    Bir tenant koleksiyonunun var olduğundan emin olur; yoksa REFERANS bir
    koleksiyonun vektör şemasını (dense boyut/mesafe metriği, sparse
    yapılandırması) birebir kopyalayarak oluşturur.

    Boyut/metrik gibi değerler burada SABİT OLARAK YAZILMAZ (DRY) — embedding
    modeli ileride değişse bile bu sınıfın güncellenmesi gerekmez, referans
    koleksiyondan otomatik okunur.
    """

    def __init__(self, client: QdrantClient, reference_collection: str):
        self._client = client
        self._reference_collection = reference_collection

    def ensure_exists(self, collection_name: str) -> bool:
        """
        Koleksiyon zaten varsa False döner (işlem yapılmadı).
        Yoksa referans şemayla oluşturur ve True döner.
        """
        if self._client.collection_exists(collection_name):
            return False

        if not self._client.collection_exists(self._reference_collection):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    f"Referans koleksiyon '{self._reference_collection}' "
                    "bulunamadı; yeni tenant koleksiyonu için vektör şeması "
                    "belirlenemiyor."
                ),
            )

        reference = self._client.get_collection(self._reference_collection)
        vectors_config = reference.config.params.vectors
        sparse_config = reference.config.params.sparse_vectors

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
        )
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Çözümleme — tüm çağıranların kullanacağı TEK ortak yol
# ══════════════════════════════════════════════════════════════════════════════
def resolve_tenant_collection(
    registry: TenantRegistry, musteri_id: Optional[int]
) -> str:
    """
    UserContext.musteri_id veya ServiceIdentity.musteri_id'den (bkz. auth.py)
    koleksiyon adına giden TEK ortak yol. Her iki durumda da musteri_id,
    imzası doğrulanmış bir JWT'den gelir — hiçbir zaman ham bir istek
    parametresinden okunmaz (bkz. api.py: yönetim uçlarında sirket_id/
    musteri_id artık bir sorgu parametresi DEĞİLDİR).

    musteri_id yoksa 400 fırlatır — tenant belirlenemeden hiçbir sorgu/işlem
    çalıştırılamaz (fail-closed; sessizce "varsayılan" bir koleksiyona
    düşülmez).
    """
    if musteri_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Tenant (müşteri) belirlenemedi: musteri_id eksik. Bu alan "
                "JWT'nin kendisinde bulunmalıdır (kullanıcı token'ında "
                "user_context.musteri_id, servis token'ında musteri_id claim'i)."
            ),
        )
    try:
        return registry.collection_name(musteri_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))