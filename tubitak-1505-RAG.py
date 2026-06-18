"""
Bilimp AI – Kurumsal Belge Asistanı
SaaS / B2B Light-Mode | ABAC Yetkilendirme
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os, time, json, hashlib, tempfile
import requests, torch
from uuid import uuid4

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from language_utils import choose_answer_language, build_language_policy_prompt, get_language_label
from abac import (
    AudiencePolicy, AudienceRule, UserContext,
    has_access, build_policy_from_ui, parse_ids, empty_rule_data, FIELD_LABELS,
    build_qdrant_abac_filter,
)
from chunker import chunk_text, chunk_pptx

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Bilimp AI Asistan",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

# ── Heavy imports (cached via @st.cache_resource) ────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from qdrant_client.http.models import (
    Distance, VectorParams, SparseVectorParams,
    Filter, FieldCondition, MatchValue, MatchAny,
)
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import pymupdf4llm
from markitdown import MarkItDown
from pptx import Presentation
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "Tubitak_Dokumanlar_Hybrid"
EMBED_MODEL     = "ytu-ce-cosmos/turkish-e5-large"
REGISTRY_FILE   = "belge_kayitlari.json"

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — CSS INJECTION
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — CSS INJECTION
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
<style>
/* ── Variables ────────────────────────────────────────────────────────────── */
:root {
    --primary:        #1B365D;
    --primary-dark:   #0F2C59;
    --secondary:      #00B4D8;
    --secondary-soft: #E6F7FA;
    --cta:            #FF6B35;
    --cta-hover:      #FF5A22;
    --neutral-bg:     #F4F6F9;
    --card:           #FFFFFF;
    --body:           #333333;
    --muted:          #6C757D;
    --border:         #E0E7EF;
    --success:        #28A745;
    --danger:         #DC3545;
}

/* ── Global reset ────────────────────────────────────────────────────────── */
html, body, .stApp { background-color: var(--neutral-bg) !important; }
* { font-family: 'Inter', 'Segoe UI', sans-serif !important; }

/* Material Symbols ikonlarını Inter override'ından muaf tut.
   Aksi halde ikon yerine 'keyboard_double_arrow', 'arrow_down' gibi
   ligature metinleri görünür. */
[data-testid="stIconMaterial"],
span.material-icons, span.material-icons-outlined,
span.material-symbols-rounded, span.material-symbols-outlined,
.material-symbols-rounded, i.material-icons {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                 'Material Icons' !important;
}

/* Hide Streamlit chrome & Fix Sidebar Restore Button */
#MainMenu, footer { visibility: hidden; } /* header burdan kaldırıldı! */
[data-testid="stDecoration"] { display: none; }
[data-testid="stAppDeployButton"] { display: none !important; }

/* Üst barı şeffaf yapıp geri getirme okunun (chevron) görünmesini sağlıyoruz */
header[data-testid="stHeader"] {
    background: transparent !important;
}
header[data-testid="stHeader"] button {
    color: var(--primary) !important;
    background-color: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    border-right: none !important;
    box-shadow: 4px 0 16px rgba(0,0,0,0.18) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.9) !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; margin: 8px 0 !important; }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label { color: rgba(255,255,255,0.7) !important; font-size: 11px !important; }
section[data-testid="stSidebar"] h3 { color: rgba(255,255,255,0.5) !important; font-size: 10px !important; text-transform: uppercase; letter-spacing: 1.5px; }

/* Widget etiketleri (Model, Bağlam, Yaratıcılık vb.) ve seçili değerlerin
   görünürlüğü — baseweb bazen -webkit-text-fill-color ile metni gizler. */
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] label * {
    color: rgba(255,255,255,0.85) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.85) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] *,
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Sidebar inputs */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: white !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] input[type="number"] {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
}
section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] { color: white !important; }

/* Baseweb input/select KAPSAYICILARININ beyaz zeminini koyulaştır.
   Asıl beyaz zemin bu iç div'lerden geliyordu → beyaz yazı + beyaz zemin = görünmez. */
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="base-input"],
section[data-testid="stSidebar"] [data-testid="stNumberInputContainer"],
section[data-testid="stSidebar"] [data-testid="stNumberInput"] > div {
    background-color: rgba(255,255,255,0.10) !important;
    border-color: rgba(255,255,255,0.18) !important;
}
/* Sidebar expander (📄 Chunk) içeriği koyu kalsın → Boyut/Örtüşme görünür olsun */
section[data-testid="stSidebar"] [data-testid="stExpander"],
section[data-testid="stSidebar"] [data-testid="stExpander"] details,
section[data-testid="stSidebar"] [data-testid="stExpanderDetails"],
section[data-testid="stSidebar"] .streamlit-expanderContent {
    background: transparent !important;
    border-color: rgba(255,255,255,0.15) !important;
}

/* ── Nav buttons ─────────────────────────────────────────────────────────── */
.nav-btn button {
    background: transparent !important;
    border: none !important;
    color: rgba(255,255,255,0.75) !important;
    text-align: left !important;
    width: 100% !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    margin: 1px 0 !important;
}
.nav-btn button:hover {
    background: rgba(255,255,255,0.1) !important;
    color: white !important;
}
.nav-btn-active button {
    background: rgba(0,180,216,0.25) !important;
    color: white !important;
    border-left: 3px solid var(--secondary) !important;
    font-weight: 700 !important;
}

/* ── Main content area ───────────────────────────────────────────────────── */
.main .block-container {
    padding: 24px 32px 32px !important;
    max-width: 1400px !important;
    background-color: var(--neutral-bg) !important;
}

/* ── Cards ───────────────────────────────────────────────────────────────── */
.bilimp-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 16px;
}
.bilimp-card-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}

/* ── Page titles ─────────────────────────────────────────────────────────── */
.page-title {
    font-size: 22px;
    font-weight: 700;
    color: var(--primary);
    margin: 0 0 4px 0;
}
.page-subtitle {
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 24px;
}

/* ── Identity bar (chat top) ─────────────────────────────────────────────── */
.identity-bar {
    background: var(--primary);
    color: white;
    padding: 10px 20px;
    border-radius: 10px;
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.identity-bar .id-chip {
    background: rgba(0,180,216,0.25);
    border: 1px solid rgba(0,180,216,0.5);
    color: #7DDFF0;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
.identity-bar .id-label {
    color: rgba(255,255,255,0.55);
    font-size: 11px;
    margin-right: 2px;
}

/* ── Chat messages ───────────────────────────────────────────────────────── */
.chat-wrap { display: flex; flex-direction: column; gap: 12px; padding-bottom: 8px; }

.msg-user { display: flex; justify-content: flex-end; }
.bubble-user {
    background: var(--secondary-soft);
    color: var(--body);
    padding: 12px 16px;
    border-radius: 16px 16px 4px 16px;
    max-width: 68%;
    font-size: 14px;
    line-height: 1.6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}

.msg-ai { display: flex; justify-content: flex-start; }
.bubble-ai {
    background: var(--card);
    color: var(--body);
    padding: 14px 18px 14px 20px;
    border-radius: 4px 16px 16px 16px;
    border-left: 4px solid var(--secondary);
    max-width: 78%;
    font-size: 14px;
    line-height: 1.7;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}
.bubble-ai pre, .bubble-user pre {
    background: var(--neutral-bg) !important;
    border-radius: 6px !important;
    padding: 10px !important;
    overflow-x: auto !important;
}

/* Access-deny strip (0-context guard rail) */
.access-deny-strip {
    background: #FFF0EC;
    color: var(--danger);
    border: 1.5px solid var(--danger);
    border-left: 4px solid var(--danger);
    padding: 14px 18px;
    border-radius: 4px 16px 16px 16px;
    font-size: 14px;
    font-weight: 600;
    max-width: 78%;
    box-shadow: 0 2px 10px rgba(220,53,69,0.12);
}

/* Language info strip */
.lang-strip {
    background: var(--primary);
    color: rgba(255,255,255,0.85);
    font-size: 11px;
    padding: 5px 14px;
    border-radius: 6px;
    display: inline-block;
    margin-bottom: 8px;
    letter-spacing: 0.3px;
}

/* ── ABAC Rule Builder ───────────────────────────────────────────────────── */
.rule-card {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px 12px;
    margin-bottom: 4px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
.rule-header {
    font-size: 12px;
    font-weight: 700;
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding-bottom: 10px;
    margin-bottom: 10px;
    border-bottom: 1px solid var(--border);
}
.or-divider {
    text-align: center;
    padding: 6px 0;
    position: relative;
    margin: 2px 0;
}
.or-divider span {
    background: var(--neutral-bg);
    color: var(--secondary);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    padding: 0 12px;
    position: relative;
    z-index: 1;
    border: 1px solid var(--secondary);
    border-radius: 20px;
}
.or-divider::before {
    content: '';
    position: absolute;
    top: 50%; left: 0; right: 0;
    height: 1px;
    background: var(--border);
}

/* ── Chips ───────────────────────────────────────────────────────────────── */
.chips-wrap { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.chip-item {
    display: inline-flex; align-items: center; gap: 4px;
    background: var(--secondary-soft);
    color: var(--secondary);
    border: 1px solid var(--secondary);
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
.chip-deny {
    background: #FFF0EC;
    color: var(--cta);
    border-color: var(--cta);
}
.no-access-badge {
    display: inline-block;
    background: #FFF0EC;
    color: var(--danger);
    border: 1px solid var(--danger);
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}

/* ── Documents table ─────────────────────────────────────────────────────── */
.doc-row {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    transition: box-shadow 0.2s;
}
.doc-row:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
.doc-name { font-size: 14px; font-weight: 600; color: var(--body); }
.doc-audience { font-size: 12px; color: var(--muted); margin-top: 4px; }
.doc-audience b { color: var(--secondary); }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.2s ease !important;
}
/* CTA (primary) */
[data-testid="stBaseButton-primary"] {
    background: var(--cta) !important;
    color: white !important;
    border: none !important;
    padding: 10px 22px !important;
}
[data-testid="stBaseButton-primary"]:hover {
    background: var(--cta-hover) !important;
    box-shadow: 0 4px 12px rgba(255,107,53,0.35) !important;
}
/* Secondary */
[data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    color: var(--secondary) !important;
    border: 1.5px solid var(--secondary) !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    background: var(--secondary-soft) !important;
}
/* Danger-like delete button */
.del-btn button {
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    padding: 4px 10px !important;
    font-size: 12px !important;
}
.del-btn button:hover {
    background: #FFF0EC !important;
    color: var(--danger) !important;
    border-color: var(--danger) !important;
}

/* ── Inputs in main area ─────────────────────────────────────────────────── */
.main input[type="text"],
.main input[type="number"],
.main textarea,
.main [data-baseweb="input"] {
    border-radius: 8px !important;
    border: 1.5px solid var(--border) !important;
    font-size: 13px !important;
    color: var(--body) !important;
    background: var(--card) !important;
}
.main input:focus, .main textarea:focus {
    border-color: var(--secondary) !important;
    box-shadow: 0 0 0 3px rgba(0,180,216,0.12) !important;
}

/* ── Chat input ──────────────────────────────────────────────────────────── */
[data-testid="stChatInputContainer"] {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background: var(--cta) !important;
    border-radius: 8px !important;
}
[data-testid="stChatInputSubmitButton"] button:hover {
    background: var(--cta-hover) !important;
}

/* Streamlit chat message overrides */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── Expanders ───────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--card) !important;
    border-radius: 8px !important;
    color: var(--body) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
.streamlit-expanderContent {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── Status / Spinner ────────────────────────────────────────────────────── */
[data-testid="stStatusWidget"] {
    background: var(--secondary-soft) !important;
    border: 1px solid var(--secondary) !important;
    border-radius: 8px !important;
    color: var(--secondary) !important;
}

/* ── Alerts ──────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* ── Profile card (sidebar bottom) ──────────────────────────────────────── */
.profile-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: auto;
}
.profile-card .pc-label {
    font-size: 10px;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.profile-badge {
    display: inline-flex;
    align-items: center;
    background: rgba(0,180,216,0.2);
    border: 1px solid rgba(0,180,216,0.35);
    color: #7DDFF0;
    padding: 2px 8px;
    border-radius: 16px;
    font-size: 11px;
    font-weight: 600;
    margin: 2px;
}

div[data-baseweb="popover"] *, 
div[data-baseweb="dropdown"] *, 
div[role="listbox"] * {
    color: #333333 !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] div {
    color: #FFFFFF !important;
}
/* ----------------------------------- */
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def md5(data: bytes) -> str:
    h = hashlib.md5()
    h.update(data)
    return h.hexdigest()


def load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_registry(reg: dict):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=4)


@st.cache_resource
def get_dense_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def get_sparse_embeddings():
    return FastEmbedSparse(model_name="Qdrant/bm25")


@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, check_compatibility=False)


def init_collection():
    c = get_qdrant_client()
    if not c.collection_exists(COLLECTION_NAME):
        c.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"content": VectorParams(size=1024, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )


def add_documents_to_qdrant(documents: list, file_hash: str | None = None):
    client = get_qdrant_client()
    dense = get_dense_embeddings()
    sparse = get_sparse_embeddings()
    if file_hash:
        for d in documents:
            d.metadata["file_hash"] = file_hash
    store = QdrantVectorStore(
        client=client, collection_name=COLLECTION_NAME,
        embedding=dense, vector_name="content",
        sparse_embedding=sparse, sparse_vector_name="sparse",
        retrieval_mode=RetrievalMode.HYBRID,
    )
    store.add_documents(documents=documents, ids=[str(uuid4()) for _ in documents])


def delete_by_source(source: str):
    c = get_qdrant_client()
    if c.collection_exists(COLLECTION_NAME):
        c.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(must=[
                FieldCondition(key="metadata.source", match=MatchValue(value=source))
            ]),
        )


def delete_document_globally(filename: str):
    delete_by_source(filename)
    reg = load_registry()
    reg.pop(filename, None)
    save_registry(reg)


def get_local_ollama_models() -> list[str]:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def sync_registry() -> dict:
    client = get_qdrant_client()
    registry = load_registry()
    qdrant_files: dict[str, dict] = {}

    if client.collection_exists(COLLECTION_NAME):
        scroll, _ = client.scroll(collection_name=COLLECTION_NAME, limit=2000, with_payload=True)
        for pt in scroll:
            meta = pt.payload.get("metadata", {})
            src = meta.get("source", "")
            if src and src not in qdrant_files:
                qdrant_files[src] = {
                    "audience": meta.get("audience", {}),
                    "hash": meta.get("file_hash", "unknown"),
                }

    updated = False
    for fname, info in qdrant_files.items():
        if fname not in registry:
            registry[fname] = {"hash": info["hash"], "audience": info["audience"],
                               "synced_at": str(time.time())}
            updated = True

    stale = [f for f in registry if f not in qdrant_files]
    for f in stale:
        del registry[f]
        updated = True

    if updated:
        save_registry(registry)
    return registry


# chunk_text ve chunk_pptx → chunker.py'den import edildi (dosya başında)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "page":           "chat",
        "messages":       [],
        "audience_rules": [empty_rule_data()],
        "api_key":        "",
        "llm_option":     None,
        "temperature":    0.3,
        "top_k":          5,
        "threshold":      0.40,
        "chunk_size":     2500,
        "chunk_overlap":  200,
        # UserContext fields
        "uc_sirket":      0,
        "uc_sube":        0,
        "uc_mudurlu":     0,
        "uc_birim":       0,
        "uc_grup":        "",
        "uc_bina":        0,
        "uc_pozisyon":    0,
        "uc_ptype":       0,
        "uc_kullanici":   0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — UserContext & UI components
# ══════════════════════════════════════════════════════════════════════════════
def get_user_context() -> UserContext:
    return UserContext(
        sirket_id       = st.session_state.uc_sirket    or None,
        sube_id         = st.session_state.uc_sube      or None,
        mudurluk_id     = st.session_state.uc_mudurlu   or None,
        birim_id        = st.session_state.uc_birim     or None,
        grup_ids        = parse_ids(st.session_state.uc_grup) or [],
        bina_id         = st.session_state.uc_bina      or None,
        pozisyon_id     = st.session_state.uc_pozisyon  or None,
        personel_tip_id = st.session_state.uc_ptype     or None,
        kullanici_id    = st.session_state.uc_kullanici or None,
    )


def render_chips_html(ids_text: str) -> str:
    ids = parse_ids(ids_text)
    if not ids:
        return ""
    chips = "".join(f'<span class="chip-item">{i}</span>' for i in ids)
    return f'<div class="chips-wrap">{chips}</div>'


def identity_bar_html(uc: UserContext) -> str:
    parts = []
    if uc.sirket_id:
        parts.append(f'<span class="id-label">Şirket</span><span class="id-chip">{uc.sirket_id}</span>')
    if uc.sube_id:
        parts.append(f'<span class="id-label">Şube</span><span class="id-chip">{uc.sube_id}</span>')
    if uc.mudurluk_id:
        parts.append(f'<span class="id-label">Müdürlük</span><span class="id-chip">{uc.mudurluk_id}</span>')
    if uc.birim_id:
        parts.append(f'<span class="id-label">Birim</span><span class="id-chip">{uc.birim_id}</span>')
    if uc.grup_ids:
        grp = " ".join(f'<span class="id-chip">{g}</span>' for g in uc.grup_ids)
        parts.append(f'<span class="id-label">Gruplar</span>{grp}')
    if uc.bina_id:
        parts.append(f'<span class="id-label">Bina</span><span class="id-chip">{uc.bina_id}</span>')
    if uc.pozisyon_id:
        parts.append(f'<span class="id-label">Pozisyon</span><span class="id-chip">{uc.pozisyon_id}</span>')
    if uc.personel_tip_id:
        parts.append(f'<span class="id-label">P.Tip</span><span class="id-chip">{uc.personel_tip_id}</span>')
    if uc.kullanici_id:
        parts.append(f'<span class="id-label">Kullanıcı ID</span><span class="id-chip">{uc.kullanici_id}</span>')

    if not parts:
        inner = '<span style="color:rgba(255,255,255,0.4); font-size:12px;">Kimlik bilgisi girilmemiş — sol menüden "Kimlik Girişi" seçin</span>'
    else:
        inner = " ".join(parts)
    return f'<div class="identity-bar">👤 Aktif Kimlik: {inner}</div>'


def profile_card_html(uc: UserContext) -> str:
    badges = []
    if uc.sirket_id:   badges.append(f'<span class="profile-badge">Şirket {uc.sirket_id}</span>')
    if uc.mudurluk_id: badges.append(f'<span class="profile-badge">Müd. {uc.mudurluk_id}</span>')
    for g in uc.grup_ids: badges.append(f'<span class="profile-badge">Grup {g}</span>')
    if uc.kullanici_id: badges.append(f'<span class="profile-badge">Kullanıcı {uc.kullanici_id}</span>')
    badge_str = "".join(badges) if badges else '<span style="color:rgba(255,255,255,0.3);font-size:11px;">Kimlik girilmedi</span>'
    return f"""
    <div class="profile-card">
        <div class="pc-label">Aktif Kimlik</div>
        <div>{badge_str}</div>
    </div>"""


def audience_summary_html(audience_data: dict) -> str:
    if not audience_data:
        return '<span class="no-access-badge">⚠ Erişim Yok</span>'
    try:
        policy = AudiencePolicy(**audience_data)
    except Exception:
        return '<span class="no-access-badge">⚠ Geçersiz Kural</span>'
    if policy.is_empty():
        return '<span class="no-access-badge">⚠ Erişim Yok</span>'

    rule_parts = []
    for rule in policy.rules:
        if rule.is_empty():
            continue
        data = rule.model_dump()
        field_htmls = []
        for field, label in FIELD_LABELS.items():
            val = data.get(field)
            if val:
                chips = "".join(f'<span class="chip-item">{v}</span>' for v in val)
                field_htmls.append(f'<b>{label}:</b> {chips}')
        if field_htmls:
            rule_parts.append("  ".join(field_htmls))

    return ("  <span style='color:var(--secondary);font-weight:700;'>VEYA</span>  ").join(rule_parts)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        # ── Logo ─────────────────────────────────────────────────────────────
        try:
            st.image("bilimp_logo.png", use_container_width=True)
        except Exception:
            st.markdown(
                '<div style="text-align:center;padding:16px 0 8px;font-size:20px;'
                'font-weight:800;color:white;letter-spacing:1px;">⚡ BILIMP</div>',
                unsafe_allow_html=True
            )
        st.markdown('<div style="border-bottom:1px solid rgba(255,255,255,0.1);margin:4px 0 12px;"></div>',
                    unsafe_allow_html=True)

        # ── Navigation ────────────────────────────────────────────────────────
        nav_items = [
            ("💬  Akıllı Sohbet",      "chat"),
            ("📂  Belge Yönetimi",     "documents"),
            ("👤  Kimlik Girişi",      "auth"),
        ]
        for label, page_key in nav_items:
            is_active = st.session_state.page == page_key
            css_class = "nav-btn-active" if is_active else "nav-btn"
            st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
            if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.page = page_key
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div style="border-bottom:1px solid rgba(255,255,255,0.1);margin:12px 0;"></div>',
                    unsafe_allow_html=True)
        st.markdown('<h3>🧠 Model Ayarları</h3>', unsafe_allow_html=True)

        # ── Model selection ───────────────────────────────────────────────────
        gemini_map = {
            "Gemini 2.5 Flash": "gemini-2.5-flash",
            "Gemini 3.0 Flash": "gemini-3-flash-preview",
        }
        ollama_list = get_local_ollama_models()
        model_options = list(gemini_map.keys())
        if ollama_list:
            model_options += [f"Ollama: {m}" for m in ollama_list]
        else:
            model_options.append("Ollama (Model Yok)")

        if st.session_state.llm_option not in model_options:
            st.session_state.llm_option = model_options[0]

        sel = st.selectbox("Model", model_options,
                           index=model_options.index(st.session_state.llm_option),
                           key="sb_model")
        st.session_state.llm_option = sel

        if "Gemini" in sel:
            st.session_state.api_key = st.text_input(
                "Google API Key", type="password",
                value=st.session_state.api_key, key="sb_apikey"
            )

        st.markdown('<div style="border-bottom:1px solid rgba(255,255,255,0.1);margin:8px 0;"></div>',
                    unsafe_allow_html=True)
        st.markdown('<h3>🎛️ İnce Ayarlar</h3>', unsafe_allow_html=True)

        st.session_state.temperature = st.slider(
            "Yaratıcılık", 0.0, 1.0, st.session_state.temperature, 0.1, key="sb_temp"
        )
        st.session_state.top_k = st.number_input(
            "Bağlam (Chunk)", 1, 20, st.session_state.top_k, key="sb_topk"
        )
        st.session_state.threshold = st.slider(
            "Benzerlik Eşiği", 0.0, 0.9, st.session_state.threshold, 0.05, key="sb_thresh"
        )
        with st.expander("📄 Chunk"):
            st.session_state.chunk_size = st.number_input(
                "Boyut", 500, 5000, st.session_state.chunk_size, key="sb_csize"
            )
            st.session_state.chunk_overlap = st.number_input(
                "Örtüşme", 0, 1000, st.session_state.chunk_overlap, key="sb_coverlap"
            )

        # ── Sync ──────────────────────────────────────────────────────────────
        if "registry_synced" not in st.session_state:
            sync_registry()
            st.session_state.registry_synced = True

        # ── Profile card (bottom) ─────────────────────────────────────────────
        st.markdown('<div style="margin-top:24px;"></div>', unsafe_allow_html=True)
        uc = get_user_context()
        st.markdown(profile_card_html(uc), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: KIMLIK GİRİŞİ (Auth / UserContext Monitor)
# ══════════════════════════════════════════════════════════════════════════════
MANUAL_PROFILE = "⚙️ Özel Kullanıcı (Manuel Giriş Yap)"

# Demo/sunum için hazır kurumsal personalar. Hedef kitle excelindeki belge
# kısıtlamalarına birebir oturur — tek tıkla profil değiştirme.
MOCK_PROFILES = {
    "Genel Müdür (Kurumsal Satış)": {
        "sirket": 14, "sube": 0, "mudurlu": 25, "birim": 0, "bina": 0,
        "pozisyon": 1, "ptype": 0, "kullanici": 0, "grup": "1",
    },
    "c.erdem (Analiz Destek - Özel Yetkili)": {
        "sirket": 14, "sube": 0, "mudurlu": 55, "birim": 0, "bina": 0,
        "pozisyon": 0, "ptype": 0, "kullanici": 590, "grup": "",
    },
    "Yazılım Süreç Yöneticisi (Yazılım + Yönetim)": {
        "sirket": 14, "sube": 0, "mudurlu": 13, "birim": 0, "bina": 0,
        "pozisyon": 0, "ptype": 0, "kullanici": 0, "grup": "101",
    },
    "Özlüce Kampüsü Çalışanı (Memur)": {
        "sirket": 14, "sube": 0, "mudurlu": 0, "birim": 0, "bina": 16,
        "pozisyon": 0, "ptype": 1, "kullanici": 0, "grup": "",
    },
    MANUAL_PROFILE: {
        "sirket": 0, "sube": 0, "mudurlu": 0, "birim": 0, "bina": 0,
        "pozisyon": 0, "ptype": 0, "kullanici": 0, "grup": "",
    },
}


# Form widget anahtarları (f_*) ↔ kalıcı kaynak anahtarları (uc_*) eşlemesi.
# f_* widget'lara bağlıdır ve sayfadan ayrılınca purge olur; uc_* ise düz
# session_state anahtarıdır ve sayfa değişse de kalıcıdır (kimlik korunur).
_AUTH_FIELDS = [
    ("f_sirket", "uc_sirket"), ("f_sube", "uc_sube"), ("f_mudurlu", "uc_mudurlu"),
    ("f_birim", "uc_birim"), ("f_bina", "uc_bina"), ("f_pozisyon", "uc_pozisyon"),
    ("f_ptype", "uc_ptype"), ("f_kullanici", "uc_kullanici"), ("f_grup", "uc_grup"),
]


def _apply_profile():
    """Seçilen mock persona'nın değerlerini form widget'larına (f_*) yazar.
    Manuel seçimde mevcut değerlere dokunmaz — serbest giriş için."""
    name = st.session_state.get("profile_select")
    if not name or name == MANUAL_PROFILE:
        return
    p = MOCK_PROFILES[name]
    st.session_state.f_sirket    = p["sirket"]
    st.session_state.f_sube      = p["sube"]
    st.session_state.f_mudurlu   = p["mudurlu"]
    st.session_state.f_birim     = p["birim"]
    st.session_state.f_bina      = p["bina"]
    st.session_state.f_pozisyon  = p["pozisyon"]
    st.session_state.f_ptype     = p["ptype"]
    st.session_state.f_kullanici = p["kullanici"]
    st.session_state.f_grup      = str(p["grup"])


def page_auth():
    st.markdown('<p class="page-title">👤 Kimlik Girişi</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Sorgu sırasında gönderilecek kullanıcı özniteliklerini belirleyin.</p>',
                unsafe_allow_html=True)

    # ── Hızlı Profil Değiştirici (Mock Personas Switcher) ─────────────────────
    st.selectbox(
        "🎭 Simüle Edilecek Personeli Seçin",
        list(MOCK_PROFILES.keys()),
        key="profile_select",
        on_change=_apply_profile,
        help="Demo için hazır kurumsal personalar. Manuel seçimde alanlar serbest kalır.",
    )

    # f_* widget anahtarlarını kalıcı uc_* kaynaklarından tohumla.
    # (Sayfadan ayrılınca f_* purge olur; geri gelince uc_*'tan yeniden kurulur.)
    for fk, uk in _AUTH_FIELDS:
        if fk not in st.session_state:
            st.session_state[fk] = st.session_state[uk]

    col_form, col_preview = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="bilimp-card">', unsafe_allow_html=True)
        st.markdown('<div class="bilimp-card-title">Öznitelik Değerleri</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.number_input("Şirket ID",       0, 99999, key="f_sirket")
        c2.number_input("Şube ID",         0, 99999, key="f_sube")
        c1.number_input("Müdürlük ID",     0, 99999, key="f_mudurlu")
        c2.number_input("Birim ID",        0, 99999, key="f_birim")
        c1.number_input("Bina ID",         0, 99999, key="f_bina")
        c2.number_input("Pozisyon ID",     0, 99999, key="f_pozisyon")
        c1.number_input("Personel Tip ID", 0, 99999, key="f_ptype")
        c2.number_input("Kullanıcı ID",    0, 99999, key="f_kullanici")
        st.text_input(
            "Grup ID'leri (virgülle — birden fazla grup olabilir)",
            placeholder="101, 108",
            key="f_grup",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Widget değerlerini kalıcı uc_* kaynaklarına yaz (kimlik sayfa değişse de kalsın).
        for fk, uk in _AUTH_FIELDS:
            st.session_state[uk] = st.session_state[fk]

        if st.button("💾 Kimliği Kaydet", type="primary", use_container_width=True):
            st.success("✓ Kimlik bilgileri güncellendi. Sohbet sorgularında bu bağlam kullanılacak.")

    with col_preview:
        uc = get_user_context()
        st.markdown('<div class="bilimp-card">', unsafe_allow_html=True)
        st.markdown('<div class="bilimp-card-title">Aktif Kullanıcı Bağlamı</div>', unsafe_allow_html=True)
        st.markdown(identity_bar_html(uc), unsafe_allow_html=True)
        st.markdown('<br><p style="font-size:12px;color:var(--muted);">Bu bilgiler API\'ye gönderilecek '
                    'UserContext nesnesi içeriğini oluşturur. Boş bırakılan alanlar <code>null</code> '
                    'olarak iletilir ve o öznitelik için kısıtlama uygulanmaz.</p>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:16px;">', unsafe_allow_html=True)
        st.json({
            "sirket_id":       uc.sirket_id,
            "sube_id":         uc.sube_id,
            "mudurluk_id":     uc.mudurluk_id,
            "birim_id":        uc.birim_id,
            "grup_ids":        uc.grup_ids,
            "bina_id":         uc.bina_id,
            "pozisyon_id":     uc.pozisyon_id,
            "personel_tip_id": uc.personel_tip_id,
            "kullanici_id":    uc.kullanici_id,
        })
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AKILLI SOHBET
# ══════════════════════════════════════════════════════════════════════════════
def _build_llm():
    sel = st.session_state.llm_option or ""
    temp = st.session_state.temperature
    if "Gemini" in sel:
        key = st.session_state.api_key
        if not key:
            st.error("⚠️ Google API Key girilmedi. Sol menüden API anahtarınızı girin.")
            return None
        gemini_map = {
            "Gemini 2.5 Flash": "gemini-2.5-flash",
            "Gemini 3.0 Flash": "gemini-3-flash-preview",
        }
        model_id = gemini_map.get(sel, "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(model=model_id, google_api_key=key, temperature=temp)
    elif "Ollama" in sel and "Yok" not in sel:
        model_id = sel.split(": ", 1)[1]
        return ChatOllama(model=model_id, temperature=temp)
    st.error("⚠️ Geçerli bir model seçilmedi.")
    return None


def _stream_text(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.04)


def _friendly_llm_error(e: Exception) -> str:
    """LLM çağrısı hatalarını kullanıcı dostu Türkçe mesaja çevirir."""
    msg = str(e).lower()
    if "more system memory" in msg or "out of memory" in msg or "status code: 500" in msg:
        return ("Seçili model bu makinenin belleğine sığmıyor. Daha küçük bir model "
                "deneyin (ör. `ollama pull gemma3:4b` veya `gemma3:1b`) ya da bulut "
                "modeli olarak Gemini'yi seçin.")
    if "does not support tools" in msg:
        return ("Seçili model araç çağırmayı (tool-calling) desteklemiyor. "
                "Lütfen Gemini'yi seçin veya tool destekli bir model kullanın.")
    if "connection" in msg or "refused" in msg or "max retries" in msg:
        return ("Model servisine bağlanılamadı. Ollama'nın çalıştığından "
                "(`ollama serve`) veya API anahtarının doğru olduğundan emin olun.")
    return f"Model yanıtı alınamadı: {e}"


def page_chat():
    uc = get_user_context()

    # Identity bar
    st.markdown(identity_bar_html(uc), unsafe_allow_html=True)

    # Message history (custom bubbles)
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for m in st.session_state.messages:
        if m["role"] == "user":
            st.markdown(
                f'<div class="msg-user"><div class="bubble-user">{m["content"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="msg-ai"><div class="bubble-ai">', unsafe_allow_html=True)
            st.markdown(m["content"])
            # Sources accordion
            if m.get("sources"):
                with st.expander(f"🔍 Referans Kaynaklar ({len(m['sources'])})"):
                    for i, doc in enumerate(m["sources"]):
                        score = doc.metadata.get("score", 0.0)
                        st.markdown(
                            f"**#{i+1}** &nbsp; 📄 `{doc.metadata.get('source')}` &nbsp;"
                            f"📊 Skor: `{score:.4f}`"
                        )
                        st.caption(doc.page_content[:400])
                        if i < len(m["sources"]) - 1:
                            st.divider()
            st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(
            f'<div class="msg-user"><div class="bubble-user">{prompt}</div></div>',
            unsafe_allow_html=True,
        )

        llm = _build_llm()
        if not llm:
            return

        client = get_qdrant_client()
        if not client.collection_exists(COLLECTION_NAME):
            st.error("Vektör veritabanı boş. Önce Belge Yönetimi sekmesinden belge yükleyin.")
            return

        # History for LLM
        history = []
        for msg in st.session_state.messages[-(10):]:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history.append(AIMessage(content=msg.get("content", "")))

        @tool
        def bilimp_knowledge_base(query: str):
            """Bilimp AI Asistanı'nın şirket içi bilgi bankasında arama yapar."""
            pass

        identity_str = f"Kullanıcı öznitelikleri: {uc.model_dump()}"
        system_prompt = f"""
Sen Bilimp AI Asistanısın. Yalnızca şirket içi belgelere dayanarak soru-cevap yaparsın.
{identity_str}

KURALLAR:
- Şirkete/işe dair HER TÜRLÜ olgu, liste, yemek listesi/menü, fiyat, rapor, tarih,
  prosedür, kural veya veri sorusunda 'bilimp_knowledge_base' KULLAN.
- Yalnızca selamlaşma/teşekkür veya kullanıcının kendi kimlik/bağlam bilgisi
  sorularında 'bilimp_knowledge_base' KULLANMA.
- ASLA şirkete özgü bilgi UYDURMA (yemek listesi, fiyat, tarih, prosedür vb.).
  Bilgi belgelerde yoksa "Bu bilgi şirket belgelerinde bulunmuyor." de.
- Her zaman nazik ve "siz" diliyle hitap et. Başka bir model olduğunu söyleme.
"""

        # ── Yönlendirme: KB araması mı, düz sohbet mi? ───────────────────────────
        # Gemini tool-calling destekler. Ollama/Gemma3 desteklemez (400: does not
        # support tools) → metin tabanlı sınıflandırma ile yönlendiririz.
        supports_tools = "Gemini" in (st.session_state.llm_option or "")
        use_kb = False
        ai_msg = None

        if supports_tools:
            llm_with_tools = llm.bind_tools([bilimp_knowledge_base])
            ai_msg = llm_with_tools.invoke(
                [SystemMessage(content=system_prompt)] + history[:-1] + [HumanMessage(content=prompt)]
            )
            use_kb = bool(ai_msg.tool_calls)
        else:
            router_prompt = (
                "Aşağıdaki kullanıcı sorusunu sınıflandır.\n"
                "- Soru herhangi bir bilgi, veri, liste, yemek listesi/menü, fiyat, rapor, "
                "prosedür, kural, tarih veya şirkete/işe dair SOMUT bir olgu içeriyorsa "
                "YALNIZCA 'KB' yaz.\n"
                "- YALNIZCA selamlaşma, teşekkür, küçük sohbet ya da kullanıcının kendi "
                "kimlik/bağlam bilgisi ise 'CHAT' yaz.\n"
                "Emin değilsen 'KB' yaz. Sadece tek kelime döndür (KB veya CHAT).\n\n"
                f"Soru: {prompt}"
            )
            try:
                route = llm.invoke([HumanMessage(content=router_prompt)])
                route_txt = route.content if isinstance(route.content, str) else str(route.content)
                use_kb = "KB" in route_txt.strip().upper()
            except Exception:
                use_kb = True  # emin değilsek güvenli taraf: belgelere dayan

        retrieved_docs: list = []
        final_response = ""

        if use_kb:
            with st.status("📚 Bilgi Bankası Taranıyor...", expanded=False) as s:
                dense = get_dense_embeddings()
                sparse = get_sparse_embeddings()
                store = QdrantVectorStore(
                    client=client, collection_name=COLLECTION_NAME,
                    embedding=dense, vector_name="content",
                    sparse_embedding=sparse, sparse_vector_name="sparse",
                    retrieval_mode=RetrievalMode.HYBRID,
                )
                top_k = st.session_state.top_k
                thresh = st.session_state.threshold
                qdrant_filter = build_qdrant_abac_filter(uc)
                results = store.similarity_search_with_score(prompt, k=top_k, filter=qdrant_filter)
                for doc, score in results:
                    if score >= thresh:
                        doc.metadata["score"] = score
                        retrieved_docs.append(doc)
                s.update(label=f"✓ {len(retrieved_docs)} belge getirildi", state="complete")

            # ── KESİN YETKİ GUARD-RAIL: 0-Context engelleme ───────────────────
            # Kullanıcının erişebileceği hiçbir belge yoksa LLM'i TETİKLEME.
            if not retrieved_docs:
                deny_msg = "Bu konudaki kurumsal belgelere erişim yetkiniz bulunmamaktadır."
                st.markdown(
                    f'<div class="msg-ai"><div class="access-deny-strip">⛔ {deny_msg}</div></div>',
                    unsafe_allow_html=True,
                )
                final_response = deny_msg
            else:
                context_str = "\n\n".join(d.page_content for d in retrieved_docs)
                answer_lang, q_lang, ctx_lang, lang_src = choose_answer_language(prompt, context_str)
                lang_label = get_language_label(answer_lang)

                st.markdown(
                    f'<div class="lang-strip">📚 Dokümanlardan Yanıtlanıyor — {lang_label}</div>',
                    unsafe_allow_html=True,
                )

                rag_prompt = f"""
{build_language_policy_prompt(answer_lang)}
Aşağıdaki şirket belgelerini kullanarak soruyu yanıtla.

KESİN KURALLAR (halüsinasyon önleme):
1. SADECE aşağıdaki BELGELER bölümündeki bilgilere dayan. Kendi genel bilgini
   veya tahminini ASLA kullanma.
2. Cevap belgelerde açıkça yoksa, uydurma yapma; aynen şunu söyle:
   "Bu bilgi erişebildiğiniz şirket belgelerinde bulunmuyor."
3. Tarih, liste, fiyat, isim gibi ayrıntıları yalnızca belgelerde yazıyorsa ver.
   Belgede olmayan gün/öğün/tutar EKLEME.

BELGELER:
{context_str}
"""
                rag_msgs = [SystemMessage(content=rag_prompt)] + history[:-1] + [HumanMessage(content=prompt)]
                chat_placeholder = st.empty()
                final_response = ""

                # RAG Modu Akıllı Metin Akışı
                try:
                    for chunk in llm.stream(rag_msgs):
                        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                        final_response += content
                        chat_placeholder.markdown(
                            f'<div class="msg-ai"><div class="bubble-ai">{final_response}</div></div>',
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    final_response = _friendly_llm_error(e)
                    chat_placeholder.markdown(
                        f'<div class="msg-ai"><div class="access-deny-strip">⚠️ {final_response}</div></div>',
                        unsafe_allow_html=True,
                    )

                # Referans dökümanların listelenmesi
                with st.expander(f"🔍 Referans Kaynaklar ({len(retrieved_docs)})"):
                    for i, doc in enumerate(retrieved_docs):
                        score = doc.metadata.get("score", 0.0)
                        st.markdown(f"**#{i+1}** &nbsp; 📄 `{doc.metadata.get('source')}` &nbsp; 📊 Skor: `{score:.4f}`")
                        st.caption(doc.page_content[:400])
                        if i < len(retrieved_docs) - 1:
                            st.divider()
        else:
            # Standart Sohbet Modu Akışı
            st.markdown('<div class="lang-strip">💬 Sohbet Modu</div>', unsafe_allow_html=True)
            chat_placeholder = st.empty()
            final_response = ""

            if ai_msg is not None and ai_msg.content:
                # Gemini: tool çağrısı yapmadı, içerik zaten cevap → kelime kelime akıt
                raw = ai_msg.content
                text = raw if isinstance(raw, str) else (
                    "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in raw)
                    if isinstance(raw, list) else str(raw)
                )
                for word in text.split(" "):
                    final_response += word + " "
                    chat_placeholder.markdown(
                        f'<div class="msg-ai"><div class="bubble-ai">{final_response}</div></div>',
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.02)
            else:
                # Ollama/Gemma3: taze sohbet yanıtı üret
                chat_msgs = [SystemMessage(content=system_prompt)] + history[:-1] + [HumanMessage(content=prompt)]
                try:
                    for chunk in llm.stream(chat_msgs):
                        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                        final_response += content
                        chat_placeholder.markdown(
                            f'<div class="msg-ai"><div class="bubble-ai">{final_response}</div></div>',
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    final_response = _friendly_llm_error(e)
                    chat_placeholder.markdown(
                        f'<div class="msg-ai"><div class="access-deny-strip">⚠️ {final_response}</div></div>',
                        unsafe_allow_html=True,
                    )

        # Mesajı geçmişe kaydetme ve ekranı tazeleme
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_response,
            "sources": retrieved_docs,
        })
        st.rerun()

# ── ALTTAKİ REFERANS SATIR (Bununla birleşmiş olmalı) ───────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BELGE YÖNETİMİ
# ══════════════════════════════════════════════════════════════════════════════
def _render_abac_builder():
    """ABAC Rule Builder: OR-of-ANDs block UI."""
    st.markdown(
        '<div style="font-size:13px;font-weight:700;color:var(--primary);margin-bottom:4px;">'
        '🎯 Hedef Kitle — Erişim Kuralları</div>',
        unsafe_allow_html=True,
    )
    st.caption("Her blok bir KURAL (AND). Bloklar arası VEYA (OR) mantığı. ID'leri virgülle girin.")

    rules = st.session_state.audience_rules
    field_list = list(FIELD_LABELS.items())

    for i, rule_data in enumerate(rules):
        if i > 0:
            st.markdown(
                '<div class="or-divider"><span>VEYA</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown(f'<div class="rule-card"><div class="rule-header">KURAL #{i+1} — tüm doldurulan alanlar AND mantığıyla çalışır</div>',
                    unsafe_allow_html=True)

        cols3 = st.columns(3)
        for j, (field, label) in enumerate(field_list):
            col = cols3[j % 3]
            val = col.text_input(
                label,
                value=rule_data.get(field, ""),
                key=f"rule_{i}_{field}",
                placeholder="örn: 13, 55",
                label_visibility="visible",
            )
            st.session_state.audience_rules[i][field] = val
            chips = render_chips_html(val)
            if chips:
                col.markdown(chips, unsafe_allow_html=True)

        if len(rules) > 1:
            _, del_col = st.columns([5, 1])
            with del_col:
                st.markdown('<div class="del-btn">', unsafe_allow_html=True)
                if st.button("🗑 Sil", key=f"del_rule_{i}"):
                    st.session_state.audience_rules.pop(i)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("➕ Yeni Kural Satırı Ekle (OR)", use_container_width=True):
        st.session_state.audience_rules.append(empty_rule_data())
        st.rerun()


def page_documents():
    st.markdown('<p class="page-title">📂 Belge Yönetimi</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Belge yükleyin ve ABAC erişim kurallarını tanımlayın.</p>',
                unsafe_allow_html=True)

    col_upload, col_list = st.columns([1, 1], gap="large")

    # ── LEFT: Upload + ABAC Builder ──────────────────────────────────────────
    with col_upload:
        st.markdown('<div class="bilimp-card">', unsafe_allow_html=True)
        st.markdown('<div class="bilimp-card-title">⬆️ Belge Yükle</div>', unsafe_allow_html=True)

        up_file = st.file_uploader(
            "PDF, DOCX, XLSX veya PPTX sürükleyin",
            type=["pdf", "docx", "xlsx", "pptx"],
            label_visibility="collapsed",
        )

        if up_file:
            bytes_data = up_file.getvalue()
            f_name = up_file.name
            curr_md5 = md5(bytes_data)
            client = get_qdrant_client()

            file_exists = hash_matches = False
            if client.collection_exists(COLLECTION_NAME):
                scroll, _ = client.scroll(collection_name=COLLECTION_NAME, limit=2000, with_payload=True)
                for pt in scroll:
                    meta = pt.payload.get("metadata", {})
                    if meta.get("source") == f_name:
                        file_exists = True
                        hash_matches = meta.get("file_hash") == curr_md5
                        break

            if file_exists and hash_matches:
                st.warning(f"⚠️ **{f_name}** zaten güncel durumda.")
            elif file_exists:
                st.info(f"🔄 **{f_name}** güncellenecek (içerik değişmiş).")
            else:
                st.success(f"✅ **{f_name}** sisteme hazır.")

        st.divider()
        _render_abac_builder()
        st.divider()

        if up_file:
            col_btn, _ = st.columns([2, 1])
            with col_btn:
                if st.button("🚀 Sisteme Entegre Et ve Yayınla", type="primary", use_container_width=True):
                    audience_policy = build_policy_from_ui(st.session_state.audience_rules)
                    if audience_policy.is_empty():
                        st.error("❌ En az bir geçerli erişim kuralı tanımlanmalıdır. Boş kayıt sisteme kabul edilmez.")
                    elif file_exists and hash_matches:
                        st.info("Belge zaten güncel.")
                    else:
                        audience_dict = audience_policy.model_dump()
                        with st.status("🔄 İşleniyor...", expanded=True) as s:
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=os.path.splitext(f_name)[1]
                            ) as tmp:
                                tmp.write(bytes_data)
                                tmp_path = tmp.name
                            s.write("Koleksiyon hazırlanıyor...")
                            init_collection()
                            delete_by_source(f_name)
                            s.write("Belge işleniyor...")
                            if f_name.endswith(".pptx"):
                                chunks = chunk_pptx(tmp_path, f_name, audience_dict)
                            else:
                                chunks = chunk_text(
                                    tmp_path, f_name,
                                    st.session_state.chunk_size,
                                    st.session_state.chunk_overlap,
                                    audience_dict,
                                )
                            os.unlink(tmp_path)
                            if chunks:
                                s.write(f"{len(chunks)} chunk Qdrant'a yükleniyor...")
                                add_documents_to_qdrant(chunks, file_hash=curr_md5)
                                reg = load_registry()
                                reg[f_name] = {
                                    "hash": curr_md5,
                                    "audience": audience_dict,
                                    "updated_at": str(time.time()),
                                }
                                save_registry(reg)
                                s.update(label=f"✅ {len(chunks)} chunk yüklendi!", state="complete")
                                st.toast("Belge sisteme entegre edildi!", icon="🎉")
                                time.sleep(0.8)
                                st.rerun()
                            else:
                                s.update(label="❌ Belge ayrıştırılamadı.", state="error")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT: Document list ──────────────────────────────────────────────────
    with col_list:
        st.markdown('<div class="bilimp-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="bilimp-card-title" '
            'style="background:var(--primary);color:white;margin:-20px -24px 16px;'
            'padding:12px 24px;border-radius:12px 12px 0 0;">🗂️ Sistemdeki Belgeler ve İzin Matrisleri</div>',
            unsafe_allow_html=True,
        )

        client = get_qdrant_client()
        doc_map: dict[str, dict] = {}

        if client.collection_exists(COLLECTION_NAME):
            scroll, _ = client.scroll(collection_name=COLLECTION_NAME, limit=2000, with_payload=True)
            for pt in scroll:
                meta = pt.payload.get("metadata", {})
                src = meta.get("source", "")
                if src and src not in doc_map:
                    doc_map[src] = {
                        "audience": meta.get("audience", {}),
                        "file_type": meta.get("file_type", "?"),
                    }

        if not doc_map:
            st.info("Henüz belge yüklenmemiş.")
        else:
            uc = get_user_context()
            for fname, info in doc_map.items():
                audience_data = info.get("audience", {})
                try:
                    policy = AudiencePolicy(**audience_data) if audience_data else AudiencePolicy()
                except Exception:
                    policy = AudiencePolicy()

                accessible = has_access(policy, uc)
                access_icon = "🟢" if accessible else "🔴"

                ext = info.get("file_type", "")
                icon = {"pdf": "📄", "docx": "📝", "xlsx": "📊", "pptx": "📑"}.get(ext, "📁")

                aud_html = audience_summary_html(audience_data)

                col_doc, col_del = st.columns([0.88, 0.12])
                with col_doc:
                    st.markdown(f"""
<div class="doc-row">
  <div style="flex:1;">
    <div class="doc-name">{icon} {fname} <span style="font-size:11px;">{access_icon}</span></div>
    <div class="doc-audience">{aud_html}</div>
  </div>
</div>""", unsafe_allow_html=True)
                with col_del:
                    st.markdown('<div class="del-btn">', unsafe_allow_html=True)
                    if st.button("🗑", key=f"del_{fname}", help="Belgeyi sil"):
                        delete_document_globally(fname)
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — CSS + Sidebar + Router
# ══════════════════════════════════════════════════════════════════════════════
inject_css()
render_sidebar()

page = st.session_state.get("page", "chat")
if page == "auth":
    page_auth()
elif page == "documents":
    page_documents()
else:
    page_chat()
