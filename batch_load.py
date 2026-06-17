#!/usr/bin/env python3
"""
Batch document loader with ABAC audience rules.

Processes files from Dokuman/ folder and uploads to Qdrant.
Audience rules are read from audience_config.json — no hardcoding.

Usage:
    python batch_load.py          # Load only new/changed files
    python batch_load.py --reset  # Clear collection and reload all
"""
import os, sys, json, hashlib, time, argparse, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abac import AudiencePolicy
from chunker import chunk_text, chunk_pptx

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, SparseVectorParams,
    Filter, FieldCondition, MatchValue,
)
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4

# ── Settings ─────────────────────────────────────────────────────────────────
QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "Tubitak_Dokumanlar_Hybrid"
MODEL_NAME      = "ytu-ce-cosmos/turkish-e5-large"
REGISTRY_FILE   = os.path.join(os.path.dirname(__file__), "belge_kayitlari.json")
CONFIG_FILE     = os.path.join(os.path.dirname(__file__), "audience_config.json")
DOKUMAN_BASE    = os.path.join(os.path.dirname(__file__), "Doküman")
CHUNK_SIZE      = 2500
CHUNK_OVERLAP   = 200


# ── Config loader ─────────────────────────────────────────────────────────────
def load_audience_config() -> dict[str, AudiencePolicy]:
    """audience_config.json'dan belge→AudiencePolicy haritasını yükler."""
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        raw: dict[str, dict] = json.load(f)
    return {fname: AudiencePolicy(**policy_data) for fname, policy_data in raw.items()}


# ── File discovery ────────────────────────────────────────────────────────────
def find_file(filename: str) -> str | None:
    for root, _, files in os.walk(DOKUMAN_BASE):
        if filename in files:
            return os.path.join(root, filename)
    return None


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Registry ──────────────────────────────────────────────────────────────────
def load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_registry(reg: dict):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=4)


# ── Qdrant helpers ────────────────────────────────────────────────────────────
def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Embedding model yükleniyor ({device})...")
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def init_collection(client: QdrantClient):
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"content": VectorParams(size=1024, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        print(f"  Koleksiyon '{COLLECTION_NAME}' oluşturuldu.")


def delete_by_source(client: QdrantClient, source: str):
    if client.collection_exists(COLLECTION_NAME):
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(must=[
                FieldCondition(key="metadata.source", match=MatchValue(value=source))
            ]),
        )


def upload(docs, dense_emb, sparse_emb, client: QdrantClient, file_hash: str):
    for doc in docs:
        doc.metadata["file_hash"] = file_hash
    store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=dense_emb,
        vector_name="content",
        sparse_embedding=sparse_emb,
        sparse_vector_name="sparse",
        retrieval_mode=RetrievalMode.HYBRID,
    )
    store.add_documents(documents=docs, ids=[str(uuid4()) for _ in docs])


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Koleksiyonu sıfırla ve tümünü yeniden yükle")
    args = parser.parse_args()

    print("═" * 60)
    print("  Bilimp Batch Loader — ABAC Belge Yükleme")
    print("═" * 60)

    # Konfigürasyon
    audience_map = load_audience_config()
    print(f"\n✓ audience_config.json yüklendi: {len(audience_map)} belge tanımı")

    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    print(f"✓ Qdrant bağlantısı: {QDRANT_URL}")

    if args.reset and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"✓ Koleksiyon silindi (--reset)")

    init_collection(client)
    dense_emb  = get_embeddings()
    sparse_emb = FastEmbedSparse(model_name="Qdrant/bm25")
    registry   = load_registry()

    results = {"ok": [], "skip": [], "missing": [], "error": []}

    for filename, policy in audience_map.items():
        print(f"\n→ {filename}")

        path = find_file(filename)
        if path is None:
            print(f"  ⚠ Doküman/ klasöründe bulunamadı — atlandı")
            results["missing"].append(filename)
            continue

        file_hash = md5(path)
        existing  = registry.get(filename, {})
        if existing.get("hash") == file_hash and not args.reset:
            print(f"  ✓ Güncel (hash eşleşiyor) — atlandı")
            results["skip"].append(filename)
            continue

        try:
            import tempfile
            audience_dict = policy.model_dump(exclude_none=True)
            ext = os.path.splitext(filename)[1].lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                with open(path, "rb") as src:
                    tmp.write(src.read())
                tmp_path = tmp.name

            delete_by_source(client, filename)

            chunks = (
                chunk_pptx(tmp_path, filename, audience_dict)
                if ext == ".pptx"
                else chunk_text(tmp_path, filename, CHUNK_SIZE, CHUNK_OVERLAP, audience_dict)
            )

            os.unlink(tmp_path)

            if not chunks:
                print(f"  ✗ Chunk üretilemedi")
                results["error"].append(filename)
                continue

            upload(chunks, dense_emb, sparse_emb, client, file_hash)

            registry[filename] = {
                "hash":       file_hash,
                "audience":   audience_dict,
                "chunks":     len(chunks),
                "updated_at": str(time.time()),
            }
            save_registry(registry)
            print(f"  ✓ {len(chunks)} chunk yüklendi | {policy.summary()}")
            results["ok"].append(filename)

        except Exception as e:
            print(f"  ✗ Hata: {e}")
            results["error"].append(filename)

    print("\n" + "═" * 60)
    print(f"  Tamamlandı: {len(results['ok'])} yüklendi | "
          f"{len(results['skip'])} atlandı | "
          f"{len(results['missing'])} bulunamadı | "
          f"{len(results['error'])} hata")
    if results["missing"]:
        print(f"  Bulunamayan: {results['missing']}")
    if results["error"]:
        print(f"  Hatalı: {results['error']}")
    print("═" * 60)


if __name__ == "__main__":
    main()
