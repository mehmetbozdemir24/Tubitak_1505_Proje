"""
audience_service.py — Doküman hedef kitle (audience) yönetimi servis katmanı.

Streamlit'ten ve API'den ORTAK kullanılan iş mantığı burada toplanır; böylece
tek bir doğruluk kaynağı olur.

Kapsanan maddeler:
 (10) Var olan bir dokümanın audience politikasını sonradan güncelleme.
 (11) Politikasız (audience'ı boş) doküman tespiti — monitoring.
 (12) Boş rules dizisi = "kimse göremez" durumunun kasıtlı onay gerektirmesi.

Qdrant veri modeli varsayımı (mevcut kodla uyumlu):
 - Her nokta payload'ında metadata.source = dosya adı
 - Her nokta payload'ında metadata.audience = {"rules": [...]} (ABAC politikası)
"""

from __future__ import annotations
import time
import logging
from dataclasses import dataclass
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from abac import AudiencePolicy

logger = logging.getLogger("audience_audit")


@dataclass
class OrphanedDocument:
    """(Faz 3 / madde 8) find_documents_without_audience'ın yapılandırılmış
    dönüş elemanı — düz dosya adı yerine versiyon bilgisini de taşır."""
    dokuman_id: str
    versiyon: Optional[int] = None


@dataclass
class AudienceInfo:
    """get_document_audience'ın yapılandırılmış dönüşü — politika ve
    optimistic locking için gereken audience_versiyon'un yanı sıra, dokümanın
    o anki İÇERİK sürümünü (icerik_versiyonu) de taşır.

    icerik_versiyonu eklenmesinin nedeni: PUT /documents/{id}/content
    (document_ingestion_service.update_document_content) 'beklenen_versiyon'
    talep eder ve dokümanda "güncel sürüm GET /audience ile öğrenilebilir"
    denir — ama audience_versiyon ile içerik versiyonu KASITLI OLARAK ayrı
    sayaçlardır (biri politika, diğeri içerik değişikliğini izler). Yalnızca
    audience_versiyon döndürülürse, 409 sonrası "güncel içerik sürümünü
    öğren, tekrar dene" kurtarma akışı hiç kodlanamaz. Bu alan tam olarak bu
    boşluğu kapatır."""
    policy: dict              # ham audience dict, örn. {"rules": [...]}
    audience_versiyon: int
    icerik_versiyonu: Optional[int] = None


class AudienceUpdateError(Exception):
    """Servis seviyesi hata; API katmanı bunu HTTP koduna çevirir."""

    def __init__(self, code: str, message: str):
        self.code = code          # "not_found" | "empty_rules" | "version_conflict"
        self.message = message
        super().__init__(message)


def _current_audience_version(client: QdrantClient, collection: str, source: str) -> Optional[int]:
    """
    (Faz 4 / madde 11) Bir dokümanın mevcut hedef kitle politikası sürümünü
    noktalarından birinin metadata'sından okur. Doküman yoksa None.

    Not: content 'versiyon' (Faz 2) ile audience_versiyon KASITLI OLARAK
    ayrı sayaçlardır — biri "içerik değişti", diğeri "kime açık olduğu
    değişti" anlamına gelir; ikisini karıştırmak yanlış bir optimistic
    locking sinyaline yol açar.
    """
    if not client.collection_exists(collection):
        return None
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(must=[
            FieldCondition(key="metadata.source", match=MatchValue(value=source))
        ]),
        limit=1,
        with_payload=True,
    )
    if not points:
        return None
    return (points[0].payload or {}).get("metadata", {}).get("audience_versiyon", 1)


def _points_for_source(client: QdrantClient, collection: str, source: str) -> list:
    """Belirli bir dosya adına ait tüm nokta ID'lerini toplar."""
    if not client.collection_exists(collection):
        # Tenant koleksiyonu henüz hiç provizyon edilmemiş → hiçbir doküman
        # yok demektir; çağıran (update_document_audience) bunu "not_found"
        # olarak ele alır.
        return []

    ids: list = []
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="metadata.source", match=MatchValue(value=source))
            ]),
            limit=256,
            with_payload=False,
            offset=next_page,
        )
        ids.extend(p.id for p in points)
        if next_page is None:
            break
    return ids


def update_document_audience(
    client: QdrantClient,
    collection: str,
    source: str,
    policy: AudiencePolicy,
    degistiren_kullanici_id: int,
    beklenen_audience_versiyon: int,
    allow_empty: bool = False,
) -> int:
    """
    (10) Bir dokümanın (source) TÜM parçalarındaki metadata.audience alanını
    yeni politika ile günceller.

    (Faz 4 / madde 11) Gerçek optimistic locking: beklenen_audience_versiyon,
    sistemdeki mevcut audience_versiyon ile eşleşmezse istek reddedilir.
    Bu, iki farklı Bilimp isteğinin (örn. iki yönetici aynı anda aynı
    dokümanı düzenlerse) birbirinin değişikliğini fark etmeden ezmesini önler.

    (12) policy boşsa (hiç kural yok veya tüm kurallar boş) doküman kimse
    tarafından görülemez hale gelir; bu ancak allow_empty=True ile kabul edilir.

    degistiren_kullanici_id: Bilimp arayüzünde değişikliği yapan kullanıcının
    kimliği. Denetim kaydı için loglanır.

    Dönüş: yeni audience_versiyon.
    Hatalar: AudienceUpdateError("not_found" | "empty_rules" | "version_conflict")
    """
    if policy.is_empty() and not allow_empty:
        raise AudienceUpdateError(
            "empty_rules",
            "Politika boş; bu doküman hiçbir kullanıcı tarafından görülemez "
            "hale gelir. Kasıtlıysa allow_empty=True ile onaylayın.",
        )

    point_ids = _points_for_source(client, collection, source)
    if not point_ids:
        raise AudienceUpdateError(
            "not_found", f"'{source}' adlı doküman sistemde bulunamadı."
        )

    mevcut_versiyon = _current_audience_version(client, collection, source) or 1
    if mevcut_versiyon != beklenen_audience_versiyon:
        raise AudienceUpdateError(
            "version_conflict",
            f"Beklenen hedef kitle sürümü {beklenen_audience_versiyon}, mevcut "
            f"sürüm {mevcut_versiyon}. Doküman aranızda başka biri tarafından "
            "güncellenmiş olabilir; güncel sürümü GET /audience ile kontrol "
            "edip tekrar deneyin.",
        )

    yeni_versiyon = mevcut_versiyon + 1

    # ÖNEMLİ: Qdrant'ın set_payload'ı, 'key' verildiğinde 'payload'
    # parametresinin HER ZAMAN bir sözlük olmasını ister — bu sözlüğün
    # alanları, 'key'in gösterdiği nesnenin İÇİNE birleştirilir (üstteki
    # kardeş alanları etkilemeden). Bu yüzden ham bir sayıyı (audience_versiyon
    # gibi) doğrudan payload olarak vermek geçersizdir; bunun yerine bir üst
    # seviyeye (metadata) çıkıp iki alanı TEK bir birleştirme çağrısında
    # güncelliyoruz — source/versiyon/file_hash gibi kardeş alanlar bu
    # birleştirmeden etkilenmez.
    client.set_payload(
        collection_name=collection,
        payload={"audience": policy.model_dump(), "audience_versiyon": yeni_versiyon},
        points=point_ids,
        key="metadata",
    )

    # Denetim kaydı: kim, neyi, ne zaman değiştirdi.
    logger.info(
        "hedef_kitle_guncellendi dokuman=%s degistiren_kullanici_id=%s "
        "yeni_politika=%s eski_versiyon=%d yeni_versiyon=%d "
        "guncellenen_nokta_sayisi=%d zaman=%d",
        source, degistiren_kullanici_id, policy.summary(), mevcut_versiyon,
        yeni_versiyon, len(point_ids), int(time.time()),
    )

    return yeni_versiyon


@dataclass
class BulkAudienceUpdateItem:
    """(Faz 4 / madde 12) Toplu güncellemedeki tek bir öğe."""
    dokuman_id: str
    policy: AudiencePolicy
    beklenen_audience_versiyon: int
    degistiren_kullanici_id: int
    allow_empty: bool = False


@dataclass
class BulkAudienceUpdateResult:
    """Toplu güncellemedeki tek bir öğenin sonucu — kısmi başarı desteklenir."""
    dokuman_id: str
    basarili: bool
    yeni_audience_versiyon: Optional[int] = None
    hata_kodu: Optional[str] = None
    mesaj: Optional[str] = None


def bulk_update_document_audience(
    client: QdrantClient, collection: str, items: list[BulkAudienceUpdateItem],
) -> list[BulkAudienceUpdateResult]:
    """
    (Faz 4 / madde 12) Birden fazla dokümanın hedef kitle politikasını TEK
    istekte günceller. Amaç: örn. bir müdürlük kapandığında yüzlerce
    dokümanın erişimini tek seferde değiştirmek — N ayrı HTTP isteği yerine.

    SENKRON işlenir; asenkron iş kuyruğu (Redis/Celery vb.) GEREKTİRMEZ.
    Gerekçe: her öğe update_document_audience'ın yaptığı hafif bir metadata
    yazma işlemidir (embedding YOK) — bu yüzden N öğe tek bir istek içinde
    makul sürede tamamlanır. Gerçekten çok büyük toplu işlemlerde (binlerce
    öğe) asenkron bir iş kuyruğuna geçmek gerekebilir; bu, request boyutu
    sınırıyla (bkz. api_schemas.py) şimdilik önlenmektedir.

    Bir öğenin başarısız olması (örn. version_conflict) DİĞERLERİNİ
    ENGELLEMEZ — her öğenin sonucu ayrı ayrı raporlanır (kısmi başarı).
    """
    results: list[BulkAudienceUpdateResult] = []
    for item in items:
        try:
            yeni_versiyon = update_document_audience(
                client=client, collection=collection, source=item.dokuman_id,
                policy=item.policy, degistiren_kullanici_id=item.degistiren_kullanici_id,
                beklenen_audience_versiyon=item.beklenen_audience_versiyon,
                allow_empty=item.allow_empty,
            )
            results.append(BulkAudienceUpdateResult(
                dokuman_id=item.dokuman_id, basarili=True,
                yeni_audience_versiyon=yeni_versiyon,
            ))
        except AudienceUpdateError as e:
            results.append(BulkAudienceUpdateResult(
                dokuman_id=item.dokuman_id, basarili=False,
                hata_kodu=e.code, mesaj=e.message,
            ))
    return results


def find_documents_without_audience(
    client: QdrantClient, collection: str, limit: int = 50, offset: int = 0,
) -> tuple[list[OrphanedDocument], int]:
    """
    (11) Hedef kitlesi tanımlanmamış (audience yok veya rules boş) dokümanları
    döndürür. Bu dokümanlar deny-by-default nedeniyle kimseye görünmez; sessizce
    kaybolmamaları için raporlanmaları gerekir.

    (Faz 3 / madde 8) Dönüş elemanları düz dosya adı değil, versiyon bilgisini
    de taşıyan yapılandırılmış nesnelerdir.

    (Faz 4 / madde 16) Sayfalama: Qdrant'ın kendisi "audience'ı boş olan
    doküman" için doğal bir filtre sunmadığından koleksiyonun taranması
    gerekir; ancak DÖNÜŞ istemciye her zaman [offset:offset+limit] dilimi +
    toplam sayı olarak verilir — böylece doküman sayısı büyüse bile tek
    yanıtın boyutu sınırlı kalır.

    Dönüş: (sayfa, toplam_politikasiz_dokuman_sayisi)
    """
    if not client.collection_exists(collection):
        return [], 0  # Tenant için henüz hiç doküman yok — raporlanacak bir şey yok.

    politikasiz_belgeler: dict[str, Optional[int]] = {}
    seen: set[str] = set()
    next_page = None

    while True:
        points, next_page = client.scroll(
            collection_name=collection,
            limit=512,
            with_payload=True,
            offset=next_page,
        )
        for p in points:
            meta = (p.payload or {}).get("metadata", {})
            source = meta.get("source")
            if not source or source in seen:
                continue
            seen.add(source)
            audience = meta.get("audience") or {}
            rules = audience.get("rules") or []
            has_real_rule = any(
                any(v not in (None, [], {}) for v in (r or {}).values())
                for r in rules
            )
            if not has_real_rule:
                politikasiz_belgeler[source] = meta.get("versiyon")
        if next_page is None:
            break

    tum_politikasiz = [
        OrphanedDocument(dokuman_id=k, versiyon=politikasiz_belgeler[k])
        for k in sorted(politikasiz_belgeler)
    ]
    toplam = len(tum_politikasiz)
    return tum_politikasiz[offset:offset + limit], toplam


def get_document_audience(
    client: QdrantClient, collection: str, source: str
) -> Optional[AudienceInfo]:
    """
    Bir dokümanın mevcut audience politikasını, hedef kitle sürümünü VE
    içerik sürümünü döndürür (GET endpoint'i için). audience_versiyon,
    PUT /audience'daki optimistic locking için; icerik_versiyonu ise
    PUT /content'teki optimistic locking için kullanılır (bkz. AudienceInfo
    docstring'i). Doküman yoksa None.
    """
    if not client.collection_exists(collection):
        return None  # Tenant koleksiyonu yok → doküman da yok.

    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(must=[
            FieldCondition(key="metadata.source", match=MatchValue(value=source))
        ]),
        limit=1,
        with_payload=True,
    )
    if not points:
        return None
    meta = (points[0].payload or {}).get("metadata", {})
    return AudienceInfo(
        policy=meta.get("audience", {"rules": []}),
        audience_versiyon=meta.get("audience_versiyon", 1),
        icerik_versiyonu=meta.get("versiyon"),
    )