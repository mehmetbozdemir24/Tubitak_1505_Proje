"""
migrate_to_tenant_collections.py — Tek koleksiyondan Seviye C çok-tenant
mimarisine tek seferlik geçiş aracı.

STRATEJİ
──────────────────────────────────────────────────────────────────────────────
Kaynak koleksiyondaki her doküman (aynı 'source' değerine sahip noktalar
kümesi), audience.rules içindeki sirket_ids alan(lar)ına bakılarak hedef
tenant koleksiyon(lar)ına kopyalanır:

  - Bir dokümanın kurallarında sirket_ids birden fazla şirket içeriyorsa
    (örn. [14, 18, 23]) — bu doküman KASITLI olarak birden fazla şirkete
    açık demektir (örnek veride "Eğitim Hafta1.pdf" böyleydi). Böyle bir
    doküman, Seviye C'nin fiziksel izolasyon prensibi gereği İLGİLİ TÜM
    tenant koleksiyonlarına kopyalanır (fiziksel izolasyon + çok şirkete
    açıklık aynı anda ancak kopyalamayla sağlanabilir).

  - Bir dokümanın hiçbir kuralında sirket_ids belirtilmemişse (gerçek test
    verisinin çoğunluğu bu durumda), tenant otomatik belirlenemez.
    --default-sirket-id ile verilen varsayılana atanır VE bu dokümanlar
    ayrıca raporlanır — üretimde bu listenin elle gözden geçirilmesi
    ÖNERİLİR, sessizce varsayılana bırakılmamalıdır.

Bu bir "en iyi çaba" (best-effort) migrasyon aracıdır; test/örnek veri gerçek
üretim verisi değildir. Gerçek geçişte doküman sahipliği muhtemelen Bilimp
tarafından bilinecektir (Faz 2 yükleme ucu tasarımıyla birlikte).

KULLANIM
    python migrate_to_tenant_collections.py \\
        --source Tubitak_Dokumanlar_Hybrid \\
        --default-sirket-id 14 \\
        [--dry-run]

--dry-run: Hiçbir şey yazmaz; yalnızca planı (hangi doküman hangi tenant
           koleksiyonuna gidecek) ve varsayılana düşen dokümanları raporlar.
"""

from __future__ import annotations
import argparse
import logging
import sys
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from tenancy import ConventionTenantRegistry, TenantCollectionProvisioner

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("migration")


def _collect_documents(client: QdrantClient, source: str) -> dict[str, list]:
    """source koleksiyonundaki tüm noktaları 'source' (dosya adı) bazında gruplar."""
    by_document: dict[str, list] = defaultdict(list)
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=source,
            limit=256,
            with_payload=True,
            with_vectors=True,
            offset=next_page,
        )
        for p in points:
            meta = (p.payload or {}).get("metadata", {})
            doc_name = meta.get("source", "__bilinmeyen__")
            by_document[doc_name].append(p)
        if next_page is None:
            break
    return by_document


def _target_sirket_ids(points: list, default_sirket_id: int) -> tuple[set[int], bool]:
    """
    Bir dokümanın noktalarındaki audience.rules'tan hedef sirket_id kümesini
    çıkarır. Dönüş: (sirket_id kümesi, varsayılana_düştü_mü).
    """
    found: set[int] = set()
    for p in points:
        meta = (p.payload or {}).get("metadata", {})
        rules = (meta.get("audience") or {}).get("rules") or []
        for rule in rules:
            for sid in (rule or {}).get("sirket_ids") or []:
                found.add(sid)

    if found:
        return found, False
    return {default_sirket_id}, True


def migrate(
    client: QdrantClient,
    source: str,
    default_sirket_id: int,
    dry_run: bool = True,
) -> None:
    registry = ConventionTenantRegistry()
    provisioner = TenantCollectionProvisioner(client, reference_collection=source)

    documents = _collect_documents(client, source)
    logger.info("Kaynak koleksiyonda %d benzersiz doküman bulundu.", len(documents))

    plan: dict[str, list[str]] = defaultdict(list)   # collection_name -> [doc_name, ...]
    fallback_docs: list[str] = []

    for doc_name, points in documents.items():
        sirket_ids, used_default = _target_sirket_ids(points, default_sirket_id)
        if used_default:
            fallback_docs.append(doc_name)
        for sid in sirket_ids:
            collection = registry.collection_name(sid)
            plan[collection].append(doc_name)

    logger.info("── Migrasyon Planı ──")
    for collection, doc_names in sorted(plan.items()):
        logger.info("  %s ← %d doküman: %s", collection, len(doc_names), doc_names)

    if fallback_docs:
        logger.warning(
            "── DİKKAT: %d doküman hiçbir kuralda sirket_ids belirtmiyor, "
            "varsayılan sirket_id=%d kullanıldı. ELLE GÖZDEN GEÇİRİN: %s",
            len(fallback_docs), default_sirket_id, fallback_docs,
        )

    if dry_run:
        logger.info("--dry-run aktif: hiçbir veri yazılmadı.")
        return

    for collection, doc_names in plan.items():
        created = provisioner.ensure_exists(collection)
        if created:
            logger.info("Koleksiyon oluşturuldu: %s", collection)

        points_to_write: list[PointStruct] = []
        for doc_name in doc_names:
            for p in documents[doc_name]:
                points_to_write.append(
                    PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                )

        client.upsert(collection_name=collection, points=points_to_write)
        logger.info("%s → %d nokta yazıldı.", collection, len(points_to_write))

    logger.info("Migrasyon tamamlandı.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Mevcut tek koleksiyonun adı")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument(
        "--default-sirket-id", type=int, required=True,
        help="sirket_ids belirtmeyen dokümanlar için varsayılan tenant",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = QdrantClient(url=args.qdrant_url, check_compatibility=False)
    if not client.collection_exists(args.source):
        logger.error("Kaynak koleksiyon bulunamadı: %s", args.source)
        sys.exit(1)

    migrate(client, args.source, args.default_sirket_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()