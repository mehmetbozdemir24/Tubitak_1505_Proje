"""
document_ingestion_service.py — Doküman Yaşam Döngüsü Servis Katmanı

Kapsam (Faz 2 / Teracity madde 2):
  - Doküman oluşturma (içerik + hedef kitle politikası TEK istekte)
  - İçerik güncelleme (yeni sürüm, optimistic locking ile)
  - Doküman silme

TASARIM KARARLARI
──────────────────────────────────────────────────────────────────────────────
dokuman_id sahipliği:
  Bilimp üretir ve yükleme isteğinde gönderir (dosya adı, örn. "rapor.pdf").
  Mevcut pipeline (chunker.py / batch_load.py) zaten metadata.source alanını
  bu şekilde kullanıyor ve tüm hedef kitle uçları (audience_service.py)
  dokuman_id'yi bu alanla eşleştiriyor — yeni bir ID sistemi icat edilmedi,
  var olan sözleşmeyle tutarlı kalındı. Tenant (koleksiyon) içinde benzersiz
  olmalıdır; benzersizlik farklı tenant'lar arasında ZORUNLU DEĞİLDİR (zaten
  fiziksel olarak ayrı koleksiyonlardadır).

Sürümleme:
  dokuman_id sabit kalır; her başarılı içerik güncellemesinde 'versiyon'
  alanı 1 artırılır. ESKİ SÜRÜMÜN TÜM İNDEKS NOKTALARI SİLİNİR — RAG her
  zaman güncel içerikten yanıt vermelidir; eski chunk'ların aynı anda
  indekste kalması çelişkili/güncel-olmayan yanıtlara yol açar. Geçmiş
  sürümlerin ayrı bir arşivde saklanması bu fazın kapsamı dışındadır.

  PUT /content isteği 'beklenen_versiyon' taşır (optimistic locking): mevcut
  versiyonla eşleşmezse 409 döner. Bu, madde 11'deki "409 sonrası davranış"
  kararıyla (optimistic locking, basit retry değil) tutarlıdır.

Mevcut pipeline'la ilişki:
  chunker.py (chunk_text / chunk_pptx) DOĞRUDAN kullanılır — parçalama
  mantığı burada TEKRARLANMAZ (DRY). Bu modülün sorumluluğu yalnızca:
  dosyayı geçici olarak diske yazmak (chunker path bekliyor), audience
  politikasını/sürüm bilgisini metadata'ya eklemek, embed edip Qdrant'a
  yazmak, ve eski sürümü temizlemektir.

  batch_load.py'deki dosya-tabanlı kayıt defteri (belge_kayitlari.json)
  BİLEREK kullanılmadı — API stateless olmalı, sürüm/hash bilgisi doğrudan
  Qdrant metadata'sında tutulur (tek doğruluk kaynağı, dosya sistemine
  bağımlılık yok — birden fazla API replikası aynı anda çalışabilir).
"""

from __future__ import annotations
import os
import hashlib
import tempfile
import logging
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from abac import AudiencePolicy
from chunker import chunk_text, chunk_pptx
from tenancy import TenantCollectionProvisioner

logger = logging.getLogger("document_ingestion")

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB — makul bir üst sınır

# Uzantı → chunker fonksiyonu eşlemesi (OCP: yeni format eklemek chunker.py'ye
# yeni bir chunk_xxx fonksiyonu + burada tek satır eklemek demektir).
_CHUNKERS = {
    ".pptx": lambda path, source, audience: chunk_pptx(path, source, audience),
}
_DEFAULT_CHUNK_SIZE = 2500
_DEFAULT_CHUNK_OVERLAP = 200


class DocumentIngestionError(Exception):
    """Servis seviyesi hata; API katmanı bunu HTTP koduna çevirir."""

    def __init__(self, code: str, message: str):
        self.code = code  # "already_exists" | "not_found" | "version_conflict" | "chunking_failed" | "too_large"
        self.message = message
        super().__init__(message)


@dataclass
class DocumentWriteResult:
    dokuman_id: str
    versiyon: int
    chunk_sayisi: int


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _chunk_file(file_bytes: bytes, dokuman_id: str, ext: str, audience_dict: dict) -> list:
    """Dosyayı geçici diske yazar, mevcut chunker.py pipeline'ını çağırır, temizler."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        chunk_fn = _CHUNKERS.get(ext)
        if chunk_fn:
            docs = chunk_fn(tmp_path, dokuman_id, audience_dict)
        else:
            docs = chunk_text(
                tmp_path, dokuman_id, _DEFAULT_CHUNK_SIZE, _DEFAULT_CHUNK_OVERLAP, audience_dict
            )
        return docs
    finally:
        os.unlink(tmp_path)


def _points_for_document(client: QdrantClient, collection: str, dokuman_id: str) -> list:
    if not client.collection_exists(collection):
        return []
    ids: list = []
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="metadata.source", match=MatchValue(value=dokuman_id))
            ]),
            limit=256,
            with_payload=False,
            offset=next_page,
        )
        ids.extend(p.id for p in points)
        if next_page is None:
            break
    return ids


def _current_version(client: QdrantClient, collection: str, dokuman_id: str) -> Optional[int]:
    """Bir dokümanın mevcut sürümünü, noktalarından birinin metadata'sından okur."""
    if not client.collection_exists(collection):
        return None
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(must=[
            FieldCondition(key="metadata.source", match=MatchValue(value=dokuman_id))
        ]),
        limit=1,
        with_payload=True,
    )
    if not points:
        return None
    return (points[0].payload or {}).get("metadata", {}).get("versiyon", 1)


def _embed_and_upsert(
    client: QdrantClient, collection: str, docs: list,
    dense_embeddings, sparse_embeddings, versiyon: int, file_hash: str,
    audience_versiyon: int,
) -> None:
    for doc in docs:
        doc.metadata["versiyon"] = versiyon
        doc.metadata["file_hash"] = file_hash
        doc.metadata["audience_versiyon"] = audience_versiyon

    store = QdrantVectorStore(
        client=client, collection_name=collection,
        embedding=dense_embeddings, vector_name="content",
        sparse_embedding=sparse_embeddings, sparse_vector_name="sparse",
        retrieval_mode=RetrievalMode.HYBRID,
    )
    store.add_documents(documents=docs, ids=[str(uuid4()) for _ in docs])


def _delete_points(client: QdrantClient, collection: str, point_ids: list) -> None:
    if point_ids:
        client.delete(collection_name=collection, points_selector=point_ids)


# ══════════════════════════════════════════════════════════════════════════════
# 1) Doküman Oluşturma
# ══════════════════════════════════════════════════════════════════════════════
def create_document(
    client: QdrantClient,
    collection: str,
    dokuman_id: str,
    file_bytes: bytes,
    file_ext: str,
    audience_policy: AudiencePolicy,
    dense_embeddings,
    sparse_embeddings,
    reference_collection: str,
    allow_empty: bool = False,
) -> DocumentWriteResult:
    """
    Yeni bir dokümanı, hedef kitle politikasıyla BİRLİKTE, tek istekte
    oluşturur.

    reference_collection: Bu tenant'ın koleksiyonu HENÜZ yoksa (bu müşteri
    için ilk doküman yükleniyorsa), vektör şemasını (boyut/mesafe metriği/
    sparse config) kopyalamak için kullanılacak var olan bir koleksiyonun
    adı — bkz. tenancy.TenantCollectionProvisioner. Bu olmadan QdrantVectorStore
    kurulumu, koleksiyon yok diye 404 ile çöker.

    allow_empty: audience_policy boşsa (hiç kural yok veya tüm kurallar boş)
    doküman oluşturulduğu anda hiçbir kullanıcı tarafından görülemez hale
    gelir (deny-by-default). Bu, update_document_audience'daki aynı
    korumanın (madde 12) create ucuna genişletilmiş halidir — Teracity'nin
    bulgusu: create ucunda bu koruma hiç yoktu, boş politikayla oluşturulan
    bir doküman sessizce "görünmez" kalabiliyordu.

    Hatalar:
      DocumentIngestionError("already_exists") — bu tenant'ta aynı dokuman_id zaten var
      DocumentIngestionError("empty_rules")    — politika boş ve allow_empty=False
      DocumentIngestionError("too_large")      — dosya MAX_FILE_SIZE_BYTES'ı aşıyor
      DocumentIngestionError("chunking_failed") — chunker.py hiç chunk üretemedi
    """
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise DocumentIngestionError(
            "too_large",
            f"Dosya {MAX_FILE_SIZE_BYTES // (1024*1024)} MB sınırını aşıyor.",
        )

    if audience_policy.is_empty() and not allow_empty:
        raise DocumentIngestionError(
            "empty_rules",
            "Hedef kitle politikası boş; bu doküman hiçbir kullanıcı tarafından "
            "görülemez hale gelir. Kasıtlıysa allow_empty=True ile onaylayın.",
        )

    # Tenant koleksiyonu henüz provizyon edilmemiş olabilir (bu müşteri için
    # ilk doküman). Yazma işleminden ÖNCE var olduğundan emin olunur.
    provisioner = TenantCollectionProvisioner(client, reference_collection=reference_collection)
    created = provisioner.ensure_exists(collection)
    if created:
        logger.info("tenant_koleksiyonu_olusturuldu koleksiyon=%s referans=%s",
                    collection, reference_collection)

    if _points_for_document(client, collection, dokuman_id):
        raise DocumentIngestionError(
            "already_exists",
            f"'{dokuman_id}' bu tenant'ta zaten mevcut. İçerik güncellemesi "
            f"için PUT /documents/{{dokuman_id}}/content kullanın.",
        )

    audience_dict = audience_policy.model_dump(exclude_none=True)
    docs = _chunk_file(file_bytes, dokuman_id, file_ext, audience_dict)
    if not docs:
        raise DocumentIngestionError(
            "chunking_failed",
            "Dosyadan hiçbir içerik çıkarılamadı (bozuk dosya veya desteklenmeyen format olabilir).",
        )

    _embed_and_upsert(
        client, collection, docs, dense_embeddings, sparse_embeddings,
        versiyon=1, file_hash=_md5(file_bytes), audience_versiyon=1,
    )

    logger.info(
        "dokuman_olusturuldu dokuman=%s koleksiyon=%s chunk_sayisi=%d",
        dokuman_id, collection, len(docs),
    )
    return DocumentWriteResult(dokuman_id=dokuman_id, versiyon=1, chunk_sayisi=len(docs))


# ══════════════════════════════════════════════════════════════════════════════
# 2) İçerik Güncelleme (Sürümleme)
# ══════════════════════════════════════════════════════════════════════════════
def update_document_content(
    client: QdrantClient,
    collection: str,
    dokuman_id: str,
    file_bytes: bytes,
    file_ext: str,
    beklenen_versiyon: int,
    dense_embeddings,
    sparse_embeddings,
) -> DocumentWriteResult:
    """
    Var olan bir dokümanın İÇERİĞİNİ günceller (yeni sürüm). Hedef kitle
    politikası KORUNUR — bu uç yalnızca içerikle ilgilenir, politika
    değişikliği için PUT /audience kullanılmalıdır.

    Optimistic locking: beklenen_versiyon mevcut sürümle eşleşmezse 409
    (madde 11 kararı).

    Hatalar:
      DocumentIngestionError("not_found")
      DocumentIngestionError("version_conflict")
      DocumentIngestionError("too_large" | "chunking_failed")
    """
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise DocumentIngestionError(
            "too_large",
            f"Dosya {MAX_FILE_SIZE_BYTES // (1024*1024)} MB sınırını aşıyor.",
        )

    mevcut_versiyon = _current_version(client, collection, dokuman_id)
    if mevcut_versiyon is None:
        raise DocumentIngestionError(
            "not_found", f"'{dokuman_id}' bu tenant'ta bulunamadı."
        )
    if mevcut_versiyon != beklenen_versiyon:
        raise DocumentIngestionError(
            "version_conflict",
            f"Beklenen sürüm {beklenen_versiyon}, mevcut sürüm {mevcut_versiyon}. "
            "Doküman aranızda başka biri tarafından güncellenmiş olabilir; "
            "güncel sürümü GET /audience ile kontrol edip tekrar deneyin.",
        )

    existing_points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(must=[
            FieldCondition(key="metadata.source", match=MatchValue(value=dokuman_id))
        ]),
        limit=1, with_payload=True,
    )
    existing_meta = (existing_points[0].payload or {}).get("metadata", {})
    audience_dict = existing_meta.get("audience", {})
    # İçerik değişse de hedef kitle politikası değişmiyor — audience_versiyon
    # (Faz 4 / madde 11) bu uçtan ETKİLENMEMELİ, olduğu gibi korunur.
    mevcut_audience_versiyon = existing_meta.get("audience_versiyon", 1)

    docs = _chunk_file(file_bytes, dokuman_id, file_ext, audience_dict)
    if not docs:
        raise DocumentIngestionError(
            "chunking_failed",
            "Dosyadan hiçbir içerik çıkarılamadı (bozuk dosya veya desteklenmeyen format olabilir).",
        )

    eski_nokta_idleri = _points_for_document(client, collection, dokuman_id)
    yeni_versiyon = mevcut_versiyon + 1

    _embed_and_upsert(
        client, collection, docs, dense_embeddings, sparse_embeddings,
        versiyon=yeni_versiyon, file_hash=_md5(file_bytes),
        audience_versiyon=mevcut_audience_versiyon,
    )
    _delete_points(client, collection, eski_nokta_idleri)

    logger.info(
        "dokuman_icerigi_guncellendi dokuman=%s koleksiyon=%s eski_versiyon=%d yeni_versiyon=%d",
        dokuman_id, collection, mevcut_versiyon, yeni_versiyon,
    )
    return DocumentWriteResult(dokuman_id=dokuman_id, versiyon=yeni_versiyon, chunk_sayisi=len(docs))


# ══════════════════════════════════════════════════════════════════════════════
# 3) Doküman Silme
# ══════════════════════════════════════════════════════════════════════════════
def delete_document(
    client: QdrantClient, collection: str, dokuman_id: str,
    beklenen_versiyon: int, degistiren_kullanici_id: int,
) -> int:
    """
    Bir dokümanın TÜM indeks noktalarını kalıcı olarak siler.

    Bu, geri alınamaz bir işlemdir; bu yüzden diğer yazma uçlarıyla (içerik
    güncelleme, hedef kitle güncelleme) AYNI iki korumayı taşır — Teracity'nin
    bulgusu: DELETE bu ikisinden muaftı, en yıkıcı işlem en az korumalıydı.

      1. Optimistic locking: beklenen_versiyon mevcut içerik sürümüyle
         eşleşmezse istek reddedilir (409) — yanlış/bayat bir sürüme
         dayanarak yanlışlıkla güncel bir dokümanın silinmesini önler.
      2. Denetim kaydı: degistiren_kullanici_id loglanır — "bu dokümanı
         kim, ne zaman sildi" sorusu her zaman cevaplanabilir kalır.

    Dönüş: silinen nokta sayısı.
    Hatalar:
      DocumentIngestionError("not_found")
      DocumentIngestionError("version_conflict")
    """
    mevcut_versiyon = _current_version(client, collection, dokuman_id)
    if mevcut_versiyon is None:
        raise DocumentIngestionError("not_found", f"'{dokuman_id}' bu tenant'ta bulunamadı.")
    if mevcut_versiyon != beklenen_versiyon:
        raise DocumentIngestionError(
            "version_conflict",
            f"Beklenen sürüm {beklenen_versiyon}, mevcut sürüm {mevcut_versiyon}. "
            "Doküman aranızda başka biri tarafından güncellenmiş olabilir; "
            "güncel sürümü GET /audience ile kontrol edip tekrar deneyin.",
        )

    point_ids = _points_for_document(client, collection, dokuman_id)
    _delete_points(client, collection, point_ids)
    logger.info(
        "dokuman_silindi dokuman=%s koleksiyon=%s degistiren_kullanici_id=%s nokta_sayisi=%d",
        dokuman_id, collection, degistiren_kullanici_id, len(point_ids),
    )
    return len(point_ids)