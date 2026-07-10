"""
tenancy.py — Çoklu Hesap (Multi-Tenant) İzolasyon Katmanı

TASARIM KARARI — Seviye C: Şirket Başına Ayrı Qdrant Koleksiyonu
──────────────────────────────────────────────────────────────────────────────
Önceki mimaride tüm şirketlerin dokümanları TEK bir Qdrant koleksiyonunda
duruyordu; şirketler arası ayrım yalnızca bir ABAC kural alanıydı
(AudienceRule.sirket_ids). Bu, bir kuralın eksik/hatalı yazılması veya
filtre kodundaki olası bir hatanın, bir şirketin dokümanlarını başka bir
şirkete sızdırabileceği anlamına geliyordu — kural, fiziksel bir sınırın
YERİNİ TUTMAZ.

Bu modül, her şirketin (sirket_id) kendi Qdrant koleksiyonuna sahip olduğu
fiziksel izolasyonu uygular. Bir sorgu, kendi tenant'ının koleksiyonu
DIŞINDA hiçbir veriye erişemez — bu artık bir kural ihlaliyle değil, bir
Qdrant koleksiyon adıyla garanti edilir.

Diğer 9 ABAC kategorisi (şube, müdürlük, birim, grup, bina, pozisyon,
personel tipi, kullanıcı, yaka tipi) bu mimariden ETKİLENMEZ — bunlar hâlâ
tenant'ın kendi koleksiyonu İÇİNDE, önceki gibi çalışmaya devam eder.
sirket_ids alanı artık koleksiyon içinde anlamsız olduğu için ABAC
filtresinden hariç tutulur (bkz. abac.build_qdrant_abac_filter'ın
exclude_fields parametresi); ama şema geriye dönük uyumluluk için
AudienceRule'da saklı kalır.

──────────────────────────────────────────────────────────────────────────────
SOLID Notları

  SRP  — İsimlendirme (TenantRegistry), fiziksel provizyon
         (TenantCollectionProvisioner) ve çözümleme (resolve_tenant_collection)
         ayrı sorumluluklar olarak ayrıştırıldı.
  OCP  — Yeni bir şirket eklemek hiçbir kod değişikliği gerektirmez;
         ConventionTenantRegistry sirket_id'den koleksiyon adını türetir.
  LSP  — TenantRegistry bir Protocol'dür; herhangi bir uyumlu implementasyon
         (örn. ileride bir veritabanı destekli kayıt) yerine geçebilir.
  ISP  — TenantRegistry yalnızca isimlendirme sorumluluğu taşır; provizyon
         (koleksiyon oluşturma) ayrı bir arayüzde (TenantCollectionProvisioner).
  DIP  — rag_service.py / audience_service.py çağıranları, somut Qdrant
         detaylarına değil, bu modüldeki soyutlamalara bağımlıdır.
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
    """Bir şirket kimliğini, o şirkete ait Qdrant koleksiyon adına çözümler."""

    def collection_name(self, sirket_id: int) -> str: ...


class ConventionTenantRegistry:
    """
    Sabit bir isimlendirme kuralına göre koleksiyon adı üretir
    ("tubitak1505_sirket_{id}"). Yeni bir şirket eklemek için kod
    değişikliği veya kayıt işlemi GEREKMEZ — sirket_id'den otomatik
    türetilir (OCP).
    """

    _PREFIX = "tubitak1505_sirket_"
    _VALID_SUFFIX = re.compile(r"^\d+$")

    def collection_name(self, sirket_id: int) -> str:
        if sirket_id is None or sirket_id < 0:
            raise ValueError(f"Geçersiz sirket_id: {sirket_id!r}")
        return f"{self._PREFIX}{sirket_id}"

    def sirket_id_from_collection(self, collection_name: str) -> Optional[int]:
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
    registry: TenantRegistry, sirket_id: Optional[int]
) -> str:
    """
    UserContext.sirket_id veya istek parametresinden koleksiyon adına giden
    tek ortak yol. sirket_id yoksa 400 fırlatır — tenant belirlenemeden
    hiçbir sorgu/işlem çalıştırılamaz (fail-closed; sessizce "varsayılan"
    bir koleksiyona düşülmez).
    """
    if sirket_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Tenant (şirket) belirlenemedi: sirket_id eksik. "
                "Sorgu uçları için JWT'nin user_context.sirket_id alanı, "
                "yönetim uçları için sirket_id istek parametresi zorunludur."
            ),
        )
    try:
        return registry.collection_name(sirket_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))