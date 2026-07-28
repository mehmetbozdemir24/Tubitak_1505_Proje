"""
migrate_to_tenant_collections.py — Tek koleksiyondan müşteri bazlı (v2.0)
çok-tenant mimarisine tek seferlik geçiş aracı.

STRATEJİ (v2.0 — musteri_id bazlı)
──────────────────────────────────────────────────────────────────────────────
v1'deki bu script, hedef tenant'ı dokümanların audience.rules'undaki
sirket_ids alanına bakarak OTOMATİK belirlemeye çalışıyordu. v2.0'da tenant
sınırı musteri_id'ye taşındığı için bu artık MÜMKÜN DEĞİL: musteri_id hiçbir
zaman bir AudienceRule alanı olmadı ve olmayacak (bkz. abac.py başlığı) —
yani eski verinin hiçbir kuralında "bu doküman şu müşteriye ait" bilgisi
yoktur ve olamaz; bu bilgi yalnızca YÜKLEME ANINDA (Bilimp'in hangi
müşterisinin bağlamında yüklendiği) bilinebilir.

Bu yüzden v2.0'da script BASİTLEŞTİ: kaynak koleksiyondaki TÜM dokümanlar,
operatörün belirttiği TEK bir --musteri-id hedefine taşınır. sirket_ids
alanı (varsa) kural içinde OLDUĞU GİBİ korunur — artık bir tenant tahmini
gerektirmez, çünkü sirket_ids v2.0'da zaten sıradan bir hedef kitle
özniteliğidir (tenant'tan bağımsız).

Birden fazla müşteriye ait karışık veri taşınması gerekiyorsa, bu script
HER MÜŞTERİ İÇİN AYRI AYRI, önce o müşteriye ait dokümanları başka bir
mekanizmayla (örn. Bilimp tarafından sağlanan bir eşleme listesi) ayırdıktan
sonra çalıştırılmalıdır — bu script'in kapsamı dışındadır.

KULLANIM
    python migrate_to_tenant_collections.py \\
        --source Tubitak_Dokumanlar_Hybrid \\
        --musteri-id 501 \\
        [--dry-run]

--dry-run: Hiçbir şey yazmaz; yalnızca taşınacak doküman listesini ve hedef
           koleksiyonu raporlar.
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


def migrate(
    client: QdrantClient,
    source: str,
    musteri_id: int,
    dry_run: bool = True,
) -> None:
    registry = ConventionTenantRegistry()
    provisioner = TenantCollectionProvisioner(client, reference_collection=source)
    target_collection = registry.collection_name(musteri_id)

    documents = _collect_documents(client, source)
    logger.info("Kaynak koleksiyonda %d benzersiz doküman bulundu.", len(documents))
    logger.info("── Migrasyon Planı ──")
    logger.info("  %s ← %d doküman (tümü): %s",
                target_collection, len(documents), sorted(documents.keys()))
    logger.info(
        "Not: sirket_ids gibi hedef kitle öznitelikleri OLDUĞU GİBİ korunur; "
        "v2.0'da bunlar tenant belirleme için kullanılmaz, sıradan bir "
        "erişim kısıtlamasıdır."
    )

    if dry_run:
        logger.info("--dry-run aktif: hiçbir veri yazılmadı.")
        return

    created = provisioner.ensure_exists(target_collection)
    if created:
        logger.info("Koleksiyon oluşturuldu: %s", target_collection)

    points_to_write: list[PointStruct] = [
        PointStruct(id=p.id, vector=p.vector, payload=p.payload)
        for points in documents.values()
        for p in points
    ]
    client.upsert(collection_name=target_collection, points=points_to_write)
    logger.info("%s → %d nokta yazıldı.", target_collection, len(points_to_write))
    logger.info("Migrasyon tamamlandı.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Mevcut tek koleksiyonun adı")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument(
        "--musteri-id", type=int, required=True,
        help="Kaynak koleksiyondaki TÜM dokümanların taşınacağı Bilimp müşteri kimliği",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = QdrantClient(url=args.qdrant_url, check_compatibility=False)
    if not client.collection_exists(args.source):
        logger.error("Kaynak koleksiyon bulunamadı: %s", args.source)
        sys.exit(1)

    migrate(client, args.source, args.musteri_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()