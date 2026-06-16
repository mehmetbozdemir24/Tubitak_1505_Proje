#!/usr/bin/env python3
"""
Batch document loader with ABAC audience rules.
Processes files from Dokuman/ folder and uploads to Qdrant.

Usage:
    python batch_load.py          # Load only new/changed files
    python batch_load.py --reset  # Clear collection and reload all
"""
import os, sys, json, hashlib, tempfile, time, argparse, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abac import AudiencePolicy, AudienceRule

# ── Lazy heavy imports (so linter doesn't fail if missing) ──────────────────
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, SparseVectorParams,
    Filter, FieldCondition, MatchValue,
)
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from markitdown import MarkItDown
from pptx import Presentation
import pymupdf4llm
from uuid import uuid4

# ── Settings ────────────────────────────────────────────────────────────────
QDRANT_URL       = "http://localhost:6333"
COLLECTION_NAME  = "Tubitak_Dokumanlar_Hybrid"
MODEL_NAME       = "ytu-ce-cosmos/turkish-e5-large"
REGISTRY_FILE    = os.path.join(os.path.dirname(__file__), "belge_kayitlari.json")
DOKUMAN_BASE     = os.path.join(os.path.dirname(__file__), "Doküman")
CHUNK_SIZE       = 2500
CHUNK_OVERLAP    = 200

# ── Audience rule map ────────────────────────────────────────────────────────
#   Key = exact filename (no path).
#   65242.pdf appears twice in the spec → combined into one policy.
AUDIENCE_MAP: dict[str, AudiencePolicy] = {
    # 1 – Şirket: Teracity (14), Prodes (18), EXA (23)
    "61270.pdf": AudiencePolicy(rules=[
        AudienceRule(sirket_ids=[14, 18, 23]),
    ]),
    # 2 – Kullanıcı: a.bilal (613)
    "61920.pdf": AudiencePolicy(rules=[
        AudienceRule(kullanici_ids=[613]),
    ]),
    # 3+5 – Müdürlük 55 & 38  VEYA  Bina (16,17) & PersonelTip Memur (1)
    "65242.pdf": AudiencePolicy(rules=[
        AudienceRule(mudurluk_ids=[55, 38]),
        AudienceRule(bina_ids=[16, 17], personel_tip_ids=[1]),
    ]),
    # 4 – Kullanıcı: c.erdem (590), e.kalıcı (591), d.hakli (596)
    "64573.pdf": AudiencePolicy(rules=[
        AudienceRule(kullanici_ids=[590, 591, 596]),
    ]),
    # 6 – (Şube 2 AND Grup 101)  VEYA  Şube 3
    "67842.pdf": AudiencePolicy(rules=[
        AudienceRule(sube_ids=[2], grup_ids=[101]),
        AudienceRule(sube_ids=[3]),
    ]),
    # 7 – Şirket: Teracity (14)
    "Haftalik_Yemek_Listesi.docx": AudiencePolicy(rules=[
        AudienceRule(sirket_ids=[14]),
    ]),
    # 8 – (Müdürlük 25 AND Grup 1 Genel Müdür)  VEYA  Müdürlük 26
    "Müşteri Listesi.xlsx": AudiencePolicy(rules=[
        AudienceRule(mudurluk_ids=[25], grup_ids=[1]),
        AudienceRule(mudurluk_ids=[26]),
    ]),
    # 9 – Müdürlük: Yazılım (13), Analiz Destek (55)
    "Bilimp_Fiyat_Listesi_Ornek_Sablon2.xlsx": AudiencePolicy(rules=[
        AudienceRule(mudurluk_ids=[13, 55]),
    ]),
    # 10 – Kullanıcı: c.erdem (590)
    "Bilimp_Fiyat_Listesi_Ornek_Sablon.xlsx": AudiencePolicy(rules=[
        AudienceRule(kullanici_ids=[590]),
    ]),
    # 11 – Pozisyon: Tasarım Yöneticisi (108), Tasarım Personeli (109)
    "173210.pdf": AudiencePolicy(rules=[
        AudienceRule(pozisyon_ids=[108, 109]),
    ]),
    # 12 – Şirket: Teracity (14)
    "183186.cleaned.pdf": AudiencePolicy(rules=[
        AudienceRule(sirket_ids=[14]),
    ]),
    # 13 – Müdürlük Yazılım (13) AND Grup Yönetim (101)
    "Butce_Ve_Maliyet_Raporu.docx": AudiencePolicy(rules=[
        AudienceRule(mudurluk_ids=[13], grup_ids=[101]),
    ]),
    # 14 – Three separate rules matching original Excel rows
    "Aylik_Satis_Ozet_Ornek_Sablon.xlsx": AudiencePolicy(rules=[
        AudienceRule(mudurluk_ids=[13], grup_ids=[101]),
        AudienceRule(mudurluk_ids=[55], grup_ids=[102]),
        AudienceRule(mudurluk_ids=[56]),
    ]),
    # 15 – Müdürlük Analiz Destek (55)  VEYA  Kullanıcı i.yildiz (755), c.erdem (759)
    "204084.cleaned.pdf": AudiencePolicy(rules=[
        AudienceRule(mudurluk_ids=[55]),
        AudienceRule(kullanici_ids=[755, 759]),
    ]),
}

# ── File discovery ───────────────────────────────────────────────────────────
def find_file(filename: str) -> str | None:
    """Search Doküman/ subdirectories for a file by name."""
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


# ── Registry ─────────────────────────────────────────────────────────────────
def load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_registry(reg: dict):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=4)


# ── Chunking ─────────────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    return text  # preserve as-is; add cleaning logic if needed


def chunk_pptx(path: str, source: str, audience: dict) -> list[Document]:
    prs = Presentation(path)
    docs = []
    for i, slide in enumerate(prs.slides):
        parts = []
        if slide.shapes.title and slide.shapes.title.text:
            parts.append(f"# {slide.shapes.title.text.strip()}")
        for shape in slide.shapes:
            if hasattr(shape, "text_frame") and shape.text_frame:
                parts.append(shape.text.strip())
        content = "\n\n".join(parts).strip()
        if content:
            docs.append(Document(
                page_content=content,
                metadata={"source": source, "chunk_no": i + 1,
                          "file_type": "pptx", "audience": audience},
            ))
    return docs


def chunk_text(path: str, source: str, audience: dict) -> list[Document]:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            raw = pymupdf4llm.to_markdown(path, write_images=False)
        else:
            md = MarkItDown()
            raw = md.convert(path).text_content

        clean = _clean(raw)

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")],
            strip_headers=True,
        )
        md_docs = header_splitter.split_text(clean)

        # Smart merge (orphan + small chunks)
        merged, temp = [], None
        for doc in md_docs:
            if not doc.page_content.strip():
                continue
            ctx = " > ".join(
                doc.metadata[k] for k in ["H1", "H2", "H3", "H4"]
                if doc.metadata.get(k)
            )
            if ctx:
                doc.page_content = f"**BAĞLAM:** {ctx}\n\n{doc.page_content}"
            if temp:
                if len(doc.page_content) < 100 and "|" not in doc.page_content:
                    merged.append(temp)
                    temp = doc
                else:
                    doc.page_content = f"{temp.page_content}\n\n{doc.page_content}"
                    merged.append(doc)
                    temp = None
            else:
                if len(doc.page_content) < 250 and "|" not in doc.page_content:
                    temp = doc
                else:
                    merged.append(doc)
        if temp:
            merged.append(temp)

        rec = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        final = []
        for doc in merged:
            doc.metadata.update({"source": source, "file_type": ext.lstrip("."), "audience": audience})
            final.extend(rec.split_documents([doc]))
        return final
    except Exception as e:
        print(f"  ✗ chunk error: {e}")
        return []


# ── Qdrant helpers ───────────────────────────────────────────────────────────
def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading embedding model on {device}...")
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
        print(f"  Collection '{COLLECTION_NAME}' created.")


def delete_by_source(client: QdrantClient, source: str):
    if client.collection_exists(COLLECTION_NAME):
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(must=[
                FieldCondition(key="metadata.source", match=MatchValue(value=source))
            ]),
        )


def upload(docs: list[Document], dense_emb, sparse_emb, client: QdrantClient, file_hash: str):
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
    ids = [str(uuid4()) for _ in docs]
    store.add_documents(documents=docs, ids=ids)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete collection before loading")
    args = parser.parse_args()

    print("═" * 60)
    print("  Bilimp Batch Loader — ABAC Document Ingestion")
    print("═" * 60)

    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    print(f"\n✓ Connected to Qdrant at {QDRANT_URL}")

    if args.reset and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"✓ Collection '{COLLECTION_NAME}' deleted (--reset)")

    init_collection(client)
    dense_emb = get_embeddings()
    sparse_emb = FastEmbedSparse(model_name="Qdrant/bm25")
    registry = load_registry()

    results = {"ok": [], "skip": [], "missing": [], "error": []}

    for filename, policy in AUDIENCE_MAP.items():
        print(f"\n→ {filename}")

        path = find_file(filename)
        if path is None:
            print(f"  ⚠ File not found in Doküman/ — skipping")
            results["missing"].append(filename)
            continue

        file_hash = md5(path)
        existing = registry.get(filename, {})
        if existing.get("hash") == file_hash and not args.reset:
            print(f"  ✓ Already up to date (hash match) — skipping")
            results["skip"].append(filename)
            continue

        try:
            audience_dict = policy.model_dump()
            ext = os.path.splitext(filename)[1].lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                with open(path, "rb") as src:
                    tmp.write(src.read())
                tmp_path = tmp.name

            delete_by_source(client, filename)

            if ext == ".pptx":
                chunks = chunk_pptx(tmp_path, filename, audience_dict)
            else:
                chunks = chunk_text(tmp_path, filename, audience_dict)

            os.unlink(tmp_path)

            if not chunks:
                print(f"  ✗ No chunks produced")
                results["error"].append(filename)
                continue

            upload(chunks, dense_emb, sparse_emb, client, file_hash)

            registry[filename] = {
                "hash": file_hash,
                "audience": audience_dict,
                "chunks": len(chunks),
                "updated_at": str(time.time()),
            }
            save_registry(registry)
            print(f"  ✓ Uploaded {len(chunks)} chunks | {policy.summary()}")
            results["ok"].append(filename)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results["error"].append(filename)

    print("\n" + "═" * 60)
    print(f"  Done: {len(results['ok'])} loaded | {len(results['skip'])} skipped |"
          f" {len(results['missing'])} missing | {len(results['error'])} errors")
    if results["missing"]:
        print(f"  Missing files: {results['missing']}")
    if results["error"]:
        print(f"  Errors: {results['error']}")
    print("═" * 60)


if __name__ == "__main__":
    main()
