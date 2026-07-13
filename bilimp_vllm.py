"""
Bilimp AI – Kurumsal Belge Asistanı
SaaS / B2B Light-Mode | ABAC Yetkilendirme

MIMARI — TEK MODEL:
  • Sohbet + Contextual + Görsel (VLM) → google/gemma-4-12B-it, vLLM (@ :8000/v1)
    (gemma-4 Unified: encoder-free multimodal — metin ve görseli aynı model işler)
  • Bulut alternatifi (opsiyonel)       → Gemini
  • Ollama artık KULLANILMIYOR.

NOT (think sızması önlemi): vLLM `--reasoning-parser gemma4` ile başlatıldığı için
modelin düşünme çıktısı ayrı alana gider, content'e karışmaz. Yine de savunma
amaçlı tüm ingestion çıktılarında <think> blokları regex ile temizlenir —
chunk içine ASLA think metni giremez.
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os, time, json, hashlib, tempfile
import requests, torch
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from language_utils import choose_answer_language, build_language_policy_prompt, get_language_label
from abac import (
    AudiencePolicy, AudienceRule, UserContext,
    has_access, build_policy_from_ui, parse_ids, empty_rule_data, FIELD_LABELS,
    build_qdrant_abac_filter,
)
import io, re, base64
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling_core.types.doc import PictureItem

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
from langchain_openai import ChatOpenAI          # vLLM (OpenAI-uyumlu) chat backend
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
# ── HOST IP otomatik algılama (WSL bozuk-localhost çözümü) ──
# Bu WSL kurulumunda loopback (localhost/127.0.0.1) güvenilir çalışmıyor;
# WSL'in kendi IP'si ise (hostname -I) her servise sorunsuz erişiyor.
# IP her WSL yeniden başlatmasında değişebileceği için sabit yazmıyoruz,
# uygulama açılışında otomatik algılıyoruz. Bulamazsa localhost'a düşer.
import subprocess

def _detect_host_ip() -> str:
    try:
        out = subprocess.run("hostname -I", shell=True,
                             capture_output=True, text=True).stdout.split()
        if out:
            return out[0]
    except Exception:
        pass
    return "localhost"

_HOST_IP = _detect_host_ip()

QDRANT_URL      = f"http://{_HOST_IP}:6333"
COLLECTION_NAME = "Tubitak_Dokumanlar_Hybrid"
EMBED_MODEL     = "ytu-ce-cosmos/turkish-e5-large"
REGISTRY_FILE   = "belge_kayitlari.json"

# ── vLLM (OpenAI-uyumlu) — TEK MODEL, her iş burada ──
# Sohbet, contextual bağlam üretimi ve görsel (VLM) açıklaması aynı modelden:
# google/gemma-4-12B-it (Unified, encoder-free multimodal).
# vLLM auth istemiyor; api_key sadece istemcinin zorunlu tuttuğu placeholder.
VLLM_BASE_URL   = f"http://{_HOST_IP}:8000/v1"
VLLM_API_KEY    = "EMPTY"
VLLM_MODEL_ID   = "google/gemma-4-12B-it"
VLLM_MAX_TOKENS = 2048

# ── Ingestion pipeline ayarları (Docling + VLM + Contextual) ──
OUTPUT_DIR      = "output-docling"            # md + figürler + kontrol dosyaları
VLM_MODEL       = VLLM_MODEL_ID               # görsel açıklama → vLLM gemma-4
CTX_MODEL       = VLLM_MODEL_ID               # contextual bağlam → vLLM gemma-4
                                              # (cache-key'de kullanılır; model değişince
                                              #  eski gemma3 cache'i otomatik geçersizleşir)
CTX_CACHE_FILE  = "contextual_cache.json"
CTX_DOC_LIMIT   = 12000
CTX_PARALLEL    = 8                            # vLLM'e eşzamanlı bağlam isteği sayısı
IMAGE_SCALE     = 2.0
USE_VLM_FOR_PICTURES = True
MIN_PICTURE_PX  = 80
TR_SPACING_ESIK = 15
MIN_PIECE_LEN   = 300                          # split-sonrası kırpık eşiği

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — CSS INJECTION
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ══════════════════════════════════════════════════════════════════════════
   1) TASARIM TOKEN'LARI
   ══════════════════════════════════════════════════════════════════════════ */
:root {
    --brand:          #1B365D;
    --brand-700:      #14294A;
    --brand-600:      #21406E;
    --accent:         #00B4D8;
    --accent-600:     #0092B0;
    --accent-soft:    #E6F7FA;
    --cta:            #FF6B35;
    --cta-hover:      #F1571F;

    --bg:             #F5F7FA;
    --card:           #FFFFFF;
    --card-2:         #FBFCFE;
    --border:         #E3E9F1;
    --border-strong:  #CBD5E3;

    --ink:            #1C2733;
    --ink-soft:       #55636F;
    --ink-muted:      #8A97A6;

    --success:        #1F9D57;
    --danger:         #DC3545;
    --danger-soft:    #FDECEE;

    --on-dark-soft:   rgba(255,255,255,0.74);
    --on-dark-muted:  rgba(255,255,255,0.50);
    --dark-surface:   rgba(255,255,255,0.09);
    --dark-border:    rgba(255,255,255,0.16);

    --r-sm: 8px;  --r-md: 12px;  --r-lg: 16px;  --r-pill: 999px;
    --sh-sm: 0 1px 3px rgba(16,32,64,0.06);
    --sh-md: 0 4px 14px rgba(16,32,64,0.08);
    --sh-lg: 0 10px 30px rgba(16,32,64,0.12);
    --sh-pop: 0 12px 32px rgba(16,32,64,0.18);
    --ring: 0 0 0 3px rgba(0,180,216,0.22);
}

/* ══════════════════════════════════════════════════════════════════════════
   2) GLOBAL
   ══════════════════════════════════════════════════════════════════════════ */
html, body, .stApp { background-color: var(--bg) !important; }
*, *::before, *::after { font-family: 'Inter','Segoe UI',system-ui,sans-serif !important; }
body { color: var(--ink); -webkit-font-smoothing: antialiased; }

[data-testid="stIconMaterial"],
span.material-icons, span.material-icons-outlined,
span.material-symbols-rounded, span.material-symbols-outlined,
.material-symbols-rounded, i.material-icons {
    font-family: 'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
}

#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stAppDeployButton"] { display: none !important; }

header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
header[data-testid="stHeader"] button {
    color: var(--brand) !important; background: #FFFFFF !important;
    border: 1px solid var(--border) !important; box-shadow: var(--sh-sm) !important; border-radius: var(--r-sm) !important;
}

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #C7D2E0; border-radius: var(--r-pill); border: 2px solid var(--bg); }
::-webkit-scrollbar-thumb:hover { background: #AEBCCE; }

/* ══════════════════════════════════════════════════════════════════════════
   3) SIDEBAR — KONTEYNER & TİPOGRAFİ
   ══════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--brand) 0%, var(--brand-700) 100%) !important;
    border-right: none !important; box-shadow: 6px 0 24px rgba(16,32,64,0.20) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 8px !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding: 0 14px 16px !important; }

section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p { color: var(--on-dark-soft) !important; }
section[data-testid="stSidebar"] hr { border-color: var(--dark-border) !important; margin: 10px 0 !important; }

section[data-testid="stSidebar"] h3 {
    color: var(--on-dark-muted) !important; font-size: 11px !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 1.4px; margin: 6px 0 8px !important;
}

/* Widget etiketleri: kutunun DIŞINDA, koyu zemin üzerinde → AÇIK renk */
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
section[data-testid="stSidebar"] > div label,
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div > label {
    color: var(--on-dark-soft) !important; -webkit-text-fill-color: var(--on-dark-soft) !important;
    font-size: 12px !important; font-weight: 600 !important;
}

/* ══════════════════════════════════════════════════════════════════════════
   3b) SIDEBAR GİRİŞ KUTULARI  →  BEYAZ ZEMİN + KOYU YAZI  (KONTRAST GARANTİSİ)
   ══════════════════════════════════════════════════════════════════════════ */
/* Dış kutular beyaz */
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="base-input"],
section[data-testid="stSidebar"] [data-testid="stNumberInputContainer"] {
    background: #FFFFFF !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--r-sm) !important;
    min-height: 42px !important;
    transition: border-color .15s ease, box-shadow .15s ease;
}
/* İçteki HER yüzey şeffaf → hiçbir alt div beyaz/gri kaçıramaz */
section[data-testid="stSidebar"] [data-baseweb="select"] > div *,
section[data-testid="stSidebar"] [data-baseweb="input"] *,
section[data-testid="stSidebar"] [data-testid="stNumberInputContainer"] > div,
section[data-testid="stSidebar"] [data-testid="stNumberInputContainer"] [data-baseweb="input"] {
    background-color: transparent !important;
}
/* Metin & değer & ikon KOYU (seçili değer dahil, her durumda okunur) */
section[data-testid="stSidebar"] [data-baseweb="select"] div,
section[data-testid="stSidebar"] [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-baseweb="select"] input,
section[data-testid="stSidebar"] input {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: var(--ink-soft) !important; }
section[data-testid="stSidebar"] input::placeholder {
    color: var(--ink-muted) !important; -webkit-text-fill-color: var(--ink-muted) !important;
}
/* Focus halkası */
section[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within,
section[data-testid="stSidebar"] [data-baseweb="input"]:focus-within,
section[data-testid="stSidebar"] [data-testid="stNumberInputContainer"]:focus-within {
    border-color: var(--accent) !important; box-shadow: var(--ring) !important;
}
/* number_input +/- düğmeleri (açık zeminde) */
section[data-testid="stSidebar"] [data-testid="stNumberInputStepUp"],
section[data-testid="stSidebar"] [data-testid="stNumberInputStepDown"] {
    background: #F1F4F9 !important; color: var(--ink-soft) !important;
    border-left: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInputStepUp"]:hover,
section[data-testid="stSidebar"] [data-testid="stNumberInputStepDown"]:hover {
    background: var(--accent) !important; color: #FFFFFF !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInputStepUp"] svg,
section[data-testid="stSidebar"] [data-testid="stNumberInputStepDown"] svg { fill: currentColor !important; }

/* ── Sidebar SLIDER (kutu yok; değer koyu zemin üstünde → AÇIK renk) ───────── */
section[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
    background: #FFFFFF !important; border: 2px solid var(--accent) !important;
}
section[data-testid="stSidebar"] [data-testid="stThumbValue"] {
    color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; font-weight: 700 !important; font-size: 12px !important;
}
section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stTickBarMax"] {
    color: var(--on-dark-muted) !important; -webkit-text-fill-color: var(--on-dark-muted) !important; font-size: 10px !important;
}

/* ── Sidebar EXPANDER (📄 Chunk): koyu translucent kutu + AÇIK yazı ───────── */
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: var(--dark-surface) !important; border: 1px solid var(--dark-border) !important;
    border-radius: var(--r-sm) !important; overflow: hidden;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary * {
    color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; font-weight: 600 !important; font-size: 12px !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { background: rgba(255,255,255,0.05) !important; }
/* Expander İÇİNDEKİ number_input etiketleri koyu zeminde → açık renk */
section[data-testid="stSidebar"] [data-testid="stExpander"] label,
section[data-testid="stSidebar"] [data-testid="stExpander"] label * {
    color: var(--on-dark-soft) !important; -webkit-text-fill-color: var(--on-dark-soft) !important;
}

/* ══════════════════════════════════════════════════════════════════════════
   4) AÇILIR MENÜ (DROPDOWN / LISTBOX) — GLOBAL, PORTALLI, DAİMA BEYAZ+KOYU
   ══════════════════════════════════════════════════════════════════════════ */
div[data-baseweb="popover"] [role="listbox"],
ul[role="listbox"] {
    background: #FFFFFF !important; border: 1px solid var(--border) !important;
    border-radius: var(--r-md) !important; box-shadow: var(--sh-pop) !important; padding: 6px !important;
}
li[role="option"] {
    background: transparent !important; border-radius: var(--r-sm) !important;
    margin: 1px 0 !important; padding: 9px 12px !important; font-size: 13px !important; transition: background .12s ease;
}
li[role="option"], li[role="option"] * {
    color: var(--brand) !important; -webkit-text-fill-color: var(--brand) !important;
}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] { background: var(--accent-soft) !important; }
li[role="option"]:hover, li[role="option"]:hover *,
li[role="option"][aria-selected="true"], li[role="option"][aria-selected="true"] * {
    color: var(--brand) !important; -webkit-text-fill-color: var(--brand) !important; font-weight: 600 !important;
}

/* ══════════════════════════════════════════════════════════════════════════
   5) SIDEBAR NAV BUTONLARI
   ══════════════════════════════════════════════════════════════════════════ */
.nav-btn button {
    background: transparent !important; border: none !important; color: var(--on-dark-soft) !important;
    text-align: left !important; width: 100% !important; padding: 11px 14px !important;
    border-radius: var(--r-sm) !important; font-size: 14px !important; font-weight: 600 !important;
    transition: all .18s ease !important; box-shadow: none !important;
}
.nav-btn button:hover { background: rgba(255,255,255,0.10) !important; color: #FFFFFF !important; }
.nav-btn-active button {
    background: linear-gradient(90deg, rgba(0,180,216,0.28), rgba(0,180,216,0.10)) !important;
    color: #FFFFFF !important; border-left: 3px solid var(--accent) !important; font-weight: 700 !important;
}

/* ══════════════════════════════════════════════════════════════════════════
   6) ANA İÇERİK — KARTLAR & BAŞLIKLAR
   ══════════════════════════════════════════════════════════════════════════ */
.main .block-container { padding: 28px 40px 40px !important; max-width: 1440px !important; background: var(--bg) !important; }

.bilimp-card {
    background: var(--card); border: 1px solid var(--border); border-radius: var(--r-lg);
    padding: 22px 26px; box-shadow: var(--sh-md); margin-bottom: 18px;
}
.bilimp-card-title {
    font-size: 12px; font-weight: 700; color: var(--brand); text-transform: uppercase; letter-spacing: 0.9px;
    margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid var(--border);
}
.page-title { font-size: 24px; font-weight: 800; color: var(--brand); margin: 0 0 4px; letter-spacing: -0.2px; }
.page-subtitle { font-size: 13.5px; color: var(--ink-soft); margin-bottom: 26px; }

/* ══════════════════════════════════════════════════════════════════════════
   7) KİMLİK ÇUBUĞU
   ══════════════════════════════════════════════════════════════════════════ */
.identity-bar {
    background: linear-gradient(90deg, var(--brand), var(--brand-600)); color: #fff;
    padding: 12px 20px; border-radius: var(--r-md); font-size: 13px; display: flex; align-items: center;
    gap: 14px; margin-bottom: 22px; flex-wrap: wrap; box-shadow: var(--sh-md);
}
.identity-bar .id-chip {
    background: rgba(0,180,216,0.22); border: 1px solid rgba(0,180,216,0.45); color: #BEEDF7;
    padding: 3px 11px; border-radius: var(--r-pill); font-size: 12px; font-weight: 700;
}
.identity-bar .id-label { color: rgba(255,255,255,0.60); font-size: 11px; margin-right: 2px; }

/* ══════════════════════════════════════════════════════════════════════════
   8) SOHBET BALONLARI
   ══════════════════════════════════════════════════════════════════════════ */
.chat-wrap { display: flex; flex-direction: column; gap: 14px; padding-bottom: 8px; }
.msg-user { display: flex; justify-content: flex-end; }
.bubble-user {
    background: var(--brand); color: #fff; padding: 12px 16px; border-radius: 16px 16px 4px 16px;
    max-width: 68%; font-size: 14px; line-height: 1.6; box-shadow: var(--sh-sm);
}
.msg-ai { display: flex; justify-content: flex-start; }
.bubble-ai {
    background: var(--card); color: var(--ink); padding: 14px 18px 14px 20px; border-radius: 4px 16px 16px 16px;
    border-left: 3px solid var(--accent); max-width: 80%; font-size: 14px; line-height: 1.7; box-shadow: var(--sh-md);
}
.bubble-ai pre, .bubble-user pre {
    background: #0F1F38 !important; color: #E6EDF5 !important; border-radius: var(--r-sm) !important;
    padding: 12px !important; overflow-x: auto !important;
}
.bubble-ai code { background: var(--accent-soft); color: var(--accent-600); padding: 1px 6px; border-radius: 5px; }

.access-deny-strip {
    background: var(--danger-soft); color: var(--danger); border: 1px solid var(--danger); border-left: 3px solid var(--danger);
    padding: 14px 18px; border-radius: 4px 16px 16px 16px; font-size: 14px; font-weight: 600; max-width: 80%; box-shadow: var(--sh-sm);
}
.lang-strip {
    background: var(--accent-soft); color: var(--accent-600); font-size: 11px; font-weight: 600; padding: 5px 14px;
    border-radius: var(--r-pill); display: inline-block; margin-bottom: 10px; letter-spacing: 0.3px; border: 1px solid rgba(0,180,216,0.30);
}

/* ══════════════════════════════════════════════════════════════════════════
   9) ABAC KURAL OLUŞTURUCU
   ══════════════════════════════════════════════════════════════════════════ */
.rule-card { background: var(--card-2); border: 1px solid var(--border); border-radius: var(--r-md); padding: 16px 18px 12px; margin-bottom: 4px; box-shadow: var(--sh-sm); }
.rule-header { font-size: 12px; font-weight: 700; color: var(--brand); text-transform: uppercase; letter-spacing: 0.7px; padding-bottom: 10px; margin-bottom: 12px; border-bottom: 1px solid var(--border); }
.or-divider { text-align: center; padding: 8px 0; position: relative; margin: 2px 0; }
.or-divider span { background: var(--bg); color: var(--accent-600); font-size: 11px; font-weight: 800; letter-spacing: 2px; padding: 2px 14px; position: relative; z-index: 1; border: 1px solid var(--accent); border-radius: var(--r-pill); }
.or-divider::before { content: ''; position: absolute; top: 50%; left: 0; right: 0; height: 1px; background: var(--border); }

/* ══════════════════════════════════════════════════════════════════════════
   10) ÇİPLER & ROZETLER
   ══════════════════════════════════════════════════════════════════════════ */
.chips-wrap { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 5px; }
.chip-item { display: inline-flex; align-items: center; gap: 4px; background: var(--accent-soft); color: var(--accent-600); border: 1px solid rgba(0,180,216,0.40); padding: 2px 11px; border-radius: var(--r-pill); font-size: 12px; font-weight: 700; }
.chip-deny { background: #FFF1EB; color: var(--cta); border-color: rgba(255,107,53,0.45); }
.no-access-badge { display: inline-block; background: var(--danger-soft); color: var(--danger); border: 1px solid rgba(220,53,69,0.40); padding: 3px 12px; border-radius: var(--r-pill); font-size: 12px; font-weight: 700; }

/* ══════════════════════════════════════════════════════════════════════════
   11) BELGE SATIRLARI
   ══════════════════════════════════════════════════════════════════════════ */
.doc-row { background: var(--card); border: 1px solid var(--border); border-radius: var(--r-md); padding: 14px 16px; margin-bottom: 10px; display: flex; align-items: flex-start; gap: 12px; transition: box-shadow .18s ease, border-color .18s ease, transform .18s ease; }
.doc-row:hover { box-shadow: var(--sh-md); border-color: var(--border-strong); transform: translateY(-1px); }
.doc-name { font-size: 14px; font-weight: 700; color: var(--ink); }
.doc-audience { font-size: 12px; color: var(--ink-soft); margin-top: 5px; line-height: 1.6; }
.doc-audience b { color: var(--accent-600); }

/* ══════════════════════════════════════════════════════════════════════════
   12) BUTONLAR
   ══════════════════════════════════════════════════════════════════════════ */
.stButton > button { border-radius: var(--r-sm) !important; font-weight: 700 !important; font-size: 13px !important; transition: all .18s ease !important; box-shadow: none !important; }
[data-testid="stBaseButton-primary"] { background: var(--cta) !important; color: #fff !important; border: none !important; padding: 11px 24px !important; box-shadow: 0 2px 8px rgba(255,107,53,0.28) !important; }
[data-testid="stBaseButton-primary"]:hover { background: var(--cta-hover) !important; box-shadow: 0 6px 18px rgba(255,107,53,0.38) !important; transform: translateY(-1px); }
[data-testid="stBaseButton-primary"]:active { transform: translateY(0); }
[data-testid="stBaseButton-secondary"] { background: #fff !important; color: var(--accent-600) !important; border: 1.5px solid var(--accent) !important; padding: 10px 22px !important; }
[data-testid="stBaseButton-secondary"]:hover { background: var(--accent-soft) !important; color: var(--accent-600) !important; }
.del-btn button { background: #fff !important; color: var(--ink-muted) !important; border: 1px solid var(--border) !important; padding: 6px 10px !important; font-size: 13px !important; }
.del-btn button:hover { background: var(--danger-soft) !important; color: var(--danger) !important; border-color: var(--danger) !important; }

/* ══════════════════════════════════════════════════════════════════════════
   13) ANA ALAN GİRİŞLERİ (açık tema) + SELECT
   ══════════════════════════════════════════════════════════════════════════ */
.main input[type="text"], .main input[type="number"], .main textarea,
.main [data-baseweb="input"], .main [data-baseweb="base-input"] {
    border-radius: var(--r-sm) !important; border: 1.5px solid var(--border) !important;
    font-size: 13.5px !important; color: var(--ink) !important; background: #fff !important;
}
.main input::placeholder, .main textarea::placeholder { color: var(--ink-muted) !important; }
.main input:focus, .main textarea:focus, .main [data-baseweb="input"]:focus-within { border-color: var(--accent) !important; box-shadow: var(--ring) !important; }
.main [data-baseweb="select"] > div { background: #fff !important; border: 1.5px solid var(--border) !important; border-radius: var(--r-sm) !important; min-height: 42px !important; }
.main [data-baseweb="select"] > div * { background-color: transparent !important; }
.main [data-baseweb="select"] > div:hover { border-color: var(--border-strong) !important; }
.main [data-baseweb="select"] > div:focus-within { border-color: var(--accent) !important; box-shadow: var(--ring) !important; }
.main [data-baseweb="select"] div, .main [data-baseweb="select"] span, .main [data-baseweb="select"] input { color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; }
.main [data-baseweb="select"] svg { fill: var(--ink-soft) !important; }

/* ══════════════════════════════════════════════════════════════════════════
   14) SOHBET GİRİŞ KUTUSU
   ══════════════════════════════════════════════════════════════════════════ */
[data-testid="stChatInput"], [data-testid="stChatInputContainer"] {
    background: #fff !important; border: 1.5px solid var(--border) !important; border-radius: var(--r-md) !important; box-shadow: var(--sh-sm) !important;
}
[data-testid="stChatInput"]:focus-within { border-color: var(--accent) !important; box-shadow: var(--ring) !important; }
[data-testid="stChatInput"] textarea { color: var(--ink) !important; }
[data-testid="stChatInputSubmitButton"] button { background: var(--cta) !important; border-radius: var(--r-sm) !important; }
[data-testid="stChatInputSubmitButton"] button:hover { background: var(--cta-hover) !important; }
[data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 0 !important; }

/* ══════════════════════════════════════════════════════════════════════════
   15) EXPANDER (ana) / STATUS / ALERT / JSON / PROGRESS
   ══════════════════════════════════════════════════════════════════════════ */
.main [data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: var(--r-md) !important; background: #fff !important; box-shadow: var(--sh-sm) !important; overflow: hidden; }
.main [data-testid="stExpander"] summary { color: var(--ink) !important; font-weight: 600 !important; font-size: 13px !important; }
.main [data-testid="stExpander"] summary:hover { background: var(--card-2) !important; color: var(--brand) !important; }
[data-testid="stStatusWidget"], [data-testid="stStatus"] { background: var(--accent-soft) !important; border: 1px solid rgba(0,180,216,0.35) !important; border-radius: var(--r-md) !important; color: var(--accent-600) !important; }
[data-testid="stAlert"] { border-radius: var(--r-md) !important; border: 1px solid var(--border) !important; }
.stAlert p { font-size: 13.5px !important; }
.main [data-testid="stJson"] { background: #0F1F38 !important; border-radius: var(--r-md) !important; padding: 12px !important; border: 1px solid var(--brand-600) !important; }
.stProgress > div > div > div > div { background: var(--accent) !important; }

/* ══════════════════════════════════════════════════════════════════════════
   16) PROFİL KARTI (sidebar dibi)
   ══════════════════════════════════════════════════════════════════════════ */
.profile-card { background: var(--dark-surface); border: 1px solid var(--dark-border); border-radius: var(--r-md); padding: 13px 15px; margin-top: 10px; }
.profile-card .pc-label { font-size: 10px; color: var(--on-dark-muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; font-weight: 700; }
.profile-badge { display: inline-flex; align-items: center; background: rgba(0,180,216,0.20); border: 1px solid rgba(0,180,216,0.38); color: #BEEDF7; padding: 3px 9px; border-radius: var(--r-pill); font-size: 11px; font-weight: 700; margin: 2px; }
</style>
""", unsafe_allow_html=True)



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


def get_vllm_models() -> list[str]:
    """vLLM OpenAI-uyumlu /models endpoint'inden yüklü model id'lerini döndürür.
    Sunucu kapalıysa boş liste → sidebar'da vLLM seçeneği görünmez."""
    try:
        r = requests.get(f"{VLLM_BASE_URL}/models", timeout=1)
        if r.status_code == 200:
            return [m["id"] for m in r.json().get("data", [])]
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


# ══════════════════════════════════════════════════════════════════════════════
# INGESTION PIPELINE — Docling+VLM MD → tablo-atomik chunk → contextual bağlam
# (eski chunker.py'nin yerini alır; chunk kuralları:
#  * bölüm chunk_size'a sığıyorsa BÜTÜN kalır (başlıktan başlığa)
#  * tablolar ASLA bölünmez; metin kesimi Madde sınırlarında
#  * breadcrumb her parçaya kopyalanır; kırpıklar komşusuna yapıştırılır)
# ══════════════════════════════════════════════════════════════════════════════
_HEADERS  = [("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")]
_CRUMB_RE = re.compile(r"^\*\*BAĞLAM:\*\*.*$", re.MULTILINE)
_CAPTION_RE   = re.compile(r"^(Şekil|Çizelge|Tablo|Figure|Table)\s*[-.]?\s*\d", re.IGNORECASE)
THINK_RE      = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
TR_SPACING_RE = re.compile(r"[a-zA-ZçÇğĞıİöÖşŞüÜ]\s[ıİşŞğĞçÇöÖüÜ]\s")

VLM_PROMPT = (
    "ONEMLI: Yanitin TAMAMEN TURKCE olmali. Tek bir Ingilizce cumle bile yazma.\n\n"
    "Bu bir kurumsal/resmi dokumandan alinmis bir GORSEL. Turkce cikti uret:\n"
    "- Tablo, confusion matrix veya sayisal grid ise: satir/sutun basliklariyla "
    "eksiksiz bir markdown tablosu yap. Anlamli, yapilandirilmis veri cikar. "
    "Tum degerleri guvenilir okuyamiyorsan UYDURMA, okuyabildigini ver.\n"
    "- Grafik ise: eksenleri ve degerleri markdown tablosu olarak ver.\n"
    "- Diyagram/sema ise: adimlari numarali Turkce maddelerle acikla.\n"
    "- Fotograf/logo ise: yeterli miktarda uzatmadan Turkce acikla.\n\n"
    "Aciklama, giris veya kapanis cumlesi ekleme. Sadece istenen Turkce icerigi ver. "
    "Yanitin ilk kelimesinden son kelimesine kadar Turkce olacak."
)

CTX_PROMPT = """<belge>
{doc}
</belge>

Yukarıdaki belgeden alınan bir parça:
<parca>
{chunk}
</parca>

Bu parçayı belgenin bütünü içinde konumlandıran, aramada bulunmasını kolaylaştıracak
KISA (en fazla 2 cümle) bir Türkçe bağlam cümlesi yaz. Belgenin ne olduğunu (tür/başlık/yıl)
ve bu parçanın neyi içerdiğini belirt. SADECE bağlam cümlesini yaz; giriş, açıklama veya
etiket ekleme."""


def _clean_vlm_output(text):
    if not text:
        return ""
    text = THINK_RE.sub("", text)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _call_vlm(png_b64, num_predict=4096, retries=3):
    """Görsel açıklamasını vLLM'deki gemma-4 (Unified multimodal) ile üretir.
    Çıktı _clean_vlm_output'tan geçer → <think> blokları chunk'a ASLA sızmaz."""
    payload = {
        "model": VLLM_MODEL_ID,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
                {"type": "text", "text": VLM_PROMPT},
            ],
        }],
        "max_tokens": num_predict,
        "temperature": 0.0,
    }
    for _ in range(retries):
        try:
            r = requests.post(f"{VLLM_BASE_URL}/chat/completions",
                              json=payload, timeout=1800)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            content = _clean_vlm_output(msg.get("content") or "")
            if content:
                return content
        except Exception:
            pass
        time.sleep(2)
    return None


def _turkce_bosluk_sorunu(text) -> bool:
    return len(TR_SPACING_RE.findall(text)) >= TR_SPACING_ESIK


def belge_to_md(input_file: str, output_dir: str, orijinal_ad: str | None = None) -> str:
    """Belgeyi (pdf/docx/pptx/xlsx) markdown'a çevirir; .md ve figürleri yazar."""
    ad   = orijinal_ad or os.path.basename(input_file)
    stem = os.path.splitext(ad)[0]
    crop_dir  = os.path.join(output_dir, stem + "_figures")
    output_md = os.path.join(output_dir, stem + ".md")
    os.makedirs(crop_dir, exist_ok=True)

    def _convert(ocr: bool):
        o = PdfPipelineOptions()
        o.do_table_structure = True
        o.do_formula_enrichment = True
        o.generate_picture_images = True
        o.images_scale = IMAGE_SCALE
        o.do_ocr = ocr
        o.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)
        conv = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=o)}
        )
        return conv.convert(input_file).document

    # 1. geçiş: OCR kapalı (metinli PDF'ler için hızlı)
    doc = _convert(ocr=False)

    # Görüntü-PDF tespiti: hem Docling hem pymupdf'in çıkardığı seçilebilir metin
    # yok denecek kadar azsa (taranmış/rasterize PDF), OCR'lı 2. geçiş yapılır.
    _gecici_md = doc.export_to_markdown()
    _metin_uz = len(re.sub(r"[^0-9A-Za-zçÇğĞıİöÖşŞüÜ]", "", _gecici_md))
    if input_file.lower().endswith(".pdf") and _metin_uz < 200:
        try:
            _fitz_uz = 0
            try:
                import fitz
                _fd = fitz.open(input_file)
                _fitz_uz = sum(len(p.get_text("text").strip()) for p in _fd)
                _fd.close()
            except Exception:
                pass
            if _fitz_uz < 200:   # gerçekten metinsiz -> OCR aç
                print(f"  Görüntü-PDF tespit edildi (seçilebilir metin ~{_metin_uz} kr) -> OCR'lı yeniden işleniyor...")
                doc = _convert(ocr=True)
        except Exception as e:
            print(f"  OCR geçişi başlatılamadı: {e}")

    desc_list = []
    if USE_VLM_FOR_PICTURES:
        # 1) Önce TÜM görselleri topla ve diske kaydet (hızlı, yerel iş)
        gorseller = []   # (pic_no, png_b64) — sıra korunur
        pic_no = 0
        for item, _level in doc.iterate_items():
            if not isinstance(item, PictureItem):
                continue
            pic_no += 1
            try:
                pil_img = item.get_image(doc)
                if pil_img is None or pil_img.width < MIN_PICTURE_PX or pil_img.height < MIN_PICTURE_PX:
                    continue
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                png = buf.getvalue()
                with open(os.path.join(crop_dir, f"gorsel_{pic_no:03d}.png"), "wb") as f:
                    f.write(png)
                gorseller.append((pic_no, base64.b64encode(png).decode()))
            except Exception:
                continue

        # 2) VLM açıklamalarını vLLM'e PARALEL gönder (görüntü istekleri
        #    KV cache'te ağır olduğu için 4 eşzamanlı ile sınırlı)
        descs = {}
        if gorseller:
            with ThreadPoolExecutor(max_workers=4) as ex:
                futs = {ex.submit(_call_vlm, b64): no for no, b64 in gorseller}
                for fut in as_completed(futs):
                    no = futs[fut]
                    try:
                        descs[no] = fut.result()
                    except Exception:
                        descs[no] = None

        # 3) Açıklamaları ORİJİNAL görsel sırasıyla listeye diz
        for no, _b64 in gorseller:
            body = descs.get(no) or "(VLM açıklaması alınamadı)"
            desc_list.append(
                f"\n\n**[Görsel {no}]** (`gorsel_{no:03d}.png`)\n\n"
                f"![Görsel {no}]({stem}_figures/gorsel_{no:03d}.png)\n\n{body}\n"
            )

    docling_md = doc.export_to_markdown()
    use_pymupdf = input_file.lower().endswith(".pdf") and _turkce_bosluk_sorunu(docling_md)

    if use_pymupdf:
        base_md = pymupdf4llm.to_markdown(input_file, write_images=False)
        base_md = re.sub(
            r"<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->",
            "", base_md, flags=re.DOTALL,
        )
        final_md = base_md.rstrip()
        if desc_list:
            final_md += "\n\n\n# ---- GÖRSEL AÇIKLAMALARI ----\n" + "".join(desc_list)
    else:
        parts = re.split(r"<!--\s*image\s*-->", docling_md)
        if len(parts) == 1:
            final_md = docling_md
            if desc_list:
                final_md = final_md.rstrip() + "\n\n\n# ---- GÖRSEL AÇIKLAMALARI ----\n" + "".join(desc_list)
        else:
            out = parts[0]
            for i, part in enumerate(parts[1:]):
                out += (desc_list[i] if i < len(desc_list) else "") + part
            final_md = out

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(final_md)
    return final_md


# ── tablo-farkında, başlık-tabanlı chunklama ──
def _blocks(content):
    out, cur, cur_is_table = [], [], None
    for line in content.splitlines():
        is_t = line.lstrip().startswith("|")
        if cur_is_table is None:
            cur_is_table = is_t
        if is_t != cur_is_table:
            out.append((cur_is_table, "\n".join(cur)))
            cur, cur_is_table = [], is_t
        cur.append(line)
    if cur:
        out.append((cur_is_table, "\n".join(cur)))
    return out


def _extract_crumb(content):
    m = _CRUMB_RE.search(content)
    return m.group(0) if m else ""


def _with_crumb(text, crumb):
    if crumb and not text.lstrip().startswith("**BAĞLAM:**"):
        return f"{crumb}\n\n{text}"
    return text


def _govde_uzunluk(piece):
    return len(_CRUMB_RE.sub("", piece).strip())


def _merge_small_pieces(pieces, crumb, min_len=MIN_PIECE_LEN):
    if len(pieces) <= 1:
        return pieces
    out, i = [], 0
    while i < len(pieces):
        p = pieces[i]
        kucuk = _govde_uzunluk(p) < min_len and "|" not in p
        if kucuk and i + 1 < len(pieces) and "|" not in pieces[i + 1]:
            nxt = pieces[i + 1]
            if crumb and nxt.lstrip().startswith(crumb):
                nxt = nxt.lstrip()[len(crumb):].lstrip("\n")
            pieces[i + 1] = f"{p}\n\n{nxt}"
        elif kucuk and out and "|" not in out[-1]:
            ek = p
            if crumb and ek.lstrip().startswith(crumb):
                ek = ek.lstrip()[len(crumb):].lstrip("\n")
            out[-1] = f"{out[-1]}\n\n{ek}"
        else:
            out.append(p)
        i += 1
    return out


def _split_section(content, chunk_size, chunk_overlap):
    if len(content) <= chunk_size:
        return [content]
    crumb = _extract_crumb(content)
    rec = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n**Madde", "\nMadde ", "\n\n", "\n", " ", ""],
    )

    # ── Tablo-kimlik iliştirme ────────────────────────────────────────────
    # Tablolar atomik kalır (ASLA bölünmez) — ama çıplak tablo chunk'ı neye
    # ait olduğunu bilmediği için retrieval'da bulunamıyordu (ör. Şekil 5
    # tablosu "Şekil 5 / T5 / BART" kelimelerini içermiyordu). Düzeltme:
    #  1) Tablodan hemen ÖNCE gelen kısa etiket satırları
    #     (**[Görsel N]**, ![...](...), **başlık**) tabloya taşınır.
    #  2) Tablodan hemen SONRA gelen "Şekil X: / Çizelge X: ..." altyazısı
    #     tabloya taşınır.
    # Böylece tablo chunk'ı kimliğini kendi içinde taşır → hem arama bulur
    # hem contextual bağlam doğru üretilir.
    bloklar = list(_blocks(content))

    # 1) ÖNCE: tablodan hemen sonraki TEK altyazıyı ("Şekil X: ...") kendi
    #    tablosuna iliştir. (Tek satır: ardışık iki altyazı varsa ikincisi
    #    bir SONRAKİ tabloya aittir, ona kalmalı.)
    for j in range(len(bloklar) - 1):
        if not bloklar[j][0] or bloklar[j + 1][0]:
            continue
        satirlar = bloklar[j + 1][1].lstrip("\n").split("\n")
        if satirlar:
            s = satirlar[0].strip()
            if s and _CAPTION_RE.match(s) and len(s) <= 250 and "|" not in s:
                cap = satirlar.pop(0)
                bloklar[j] = (True, bloklar[j][1].rstrip("\n") + "\n" + cap)
                bloklar[j + 1] = (False, "\n".join(satirlar))

    # 2) SONRA: tablodan önceki etiket satırlarını (**[Görsel N]**, görsel
    #    linki, kalan altyazı) tabloya çek.
    for j in range(len(bloklar)):
        if not bloklar[j][0]:
            continue
        if j == 0 or bloklar[j - 1][0]:
            continue
        onceki = bloklar[j - 1][1].rstrip("\n").split("\n")
        tasi = []
        while onceki and len(tasi) < 4:
            s = onceki[-1].strip()
            if not s:
                onceki.pop()
                continue
            etiket_mi = (s.startswith("**[Görsel") or s.startswith("![")
                         or s.startswith("**") or _CAPTION_RE.match(s))
            if etiket_mi and len(s) <= 250 and "|" not in s:
                tasi.insert(0, onceki.pop())
            else:
                break
        if tasi:
            bloklar[j - 1] = (False, "\n".join(onceki))
            bloklar[j] = (True, "\n".join(tasi) + "\n" + bloklar[j][1])

    pieces, text_buf = [], []

    def flush_text():
        buf = "\n".join(text_buf).strip()
        text_buf.clear()
        if not buf:
            return
        if len(buf) <= chunk_size:
            pieces.append(_with_crumb(buf, crumb))
        else:
            for p in rec.split_text(buf):
                p = p.strip()
                if p:
                    pieces.append(_with_crumb(p, crumb))

    for is_table, blk in bloklar:
        if is_table:
            flush_text()
            tbl = blk.strip()
            if tbl:
                pieces.append(_with_crumb(tbl, crumb))
        else:
            text_buf.append(blk)
    flush_text()
    return _merge_small_pieces(pieces, crumb)


def chunk_md(text, source, chunk_size, chunk_overlap, audience=None):
    md_docs = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS, strip_headers=True
    ).split_text(text)

    merged, temp = [], None
    for doc in md_docs:
        if not doc.page_content.strip():
            continue
        ctx = " > ".join(doc.metadata[h] for _, h in _HEADERS if doc.metadata.get(h))
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

    final = []
    for doc in merged:
        base_meta = dict(doc.metadata)
        base_meta["source"] = source
        base_meta["file_type"] = os.path.splitext(source)[1].lstrip(".") or "md"
        if audience is not None:
            base_meta["audience"] = audience
        for piece in _split_section(doc.page_content, chunk_size, chunk_overlap):
            final.append(Document(page_content=piece, metadata=dict(base_meta)))

    for i, d in enumerate(final, 1):
        d.metadata["chunk_no"] = i
        d.metadata["has_table"] = "|" in d.page_content
    return final


# ── contextual retrieval (lokal LLM, cache'li) ──
def _ctx_cache_load():
    if os.path.exists(CTX_CACHE_FILE):
        with open(CTX_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _ctx_cache_save(c):
    with open(CTX_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False)


def ensure_ctx_model():
    """Ingestion artık vLLM (gemma-4) kullanıyor → vLLM ayakta mı ve doğru model
    yüklü mü kontrol et. Değilse sessizce boş bağlamla devam ETME — hata ver."""
    try:
        r = requests.get(f"{VLLM_BASE_URL}/models", timeout=5)
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", [])]
    except Exception as e:
        raise RuntimeError(f"vLLM'e ulaşılamadı ({e}). vLLM sunucusu (:8000) çalışıyor mu?")
    if VLLM_MODEL_ID not in ids:
        raise RuntimeError(f"vLLM'de {VLLM_MODEL_ID} yüklü değil. "
                           f"Yüklü: {', '.join(ids) or '(hiç yok)'}")


def _uret_baglam(doc_text, chunk_text_):
    """Chunk için contextual bağlam cümlesini vLLM (gemma-4) ile üretir.
    Çıktı _clean_vlm_output'tan geçirilir → <think> blokları temizlenir;
    ayrıca vLLM reasoning-parser sayesinde düşünme zaten ayrı alanda kalır.
    Sonuç: chunk'a eklenen bağlam metninde think ASLA bulunmaz."""
    payload = {
        "model": VLLM_MODEL_ID,
        "messages": [{
            "role": "user",
            "content": CTX_PROMPT.format(doc=doc_text[:CTX_DOC_LIMIT], chunk=chunk_text_),
        }],
        "max_tokens": 512,
        "temperature": 0.0,
    }
    r = requests.post(f"{VLLM_BASE_URL}/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    return _clean_vlm_output(msg.get("content") or "")


def add_contextual(chunks, full_md, source, on_progress=None):
    """Contextual bağlamları vLLM'e PARALEL isteklerle üretir (CTX_PARALLEL
    eşzamanlı). vLLM continuous batching sayesinde bu, sıralı üretime göre
    kat kat hızlıdır. Cache diske TEK SEFERDE yazılır (chunk başına disk I/O
    /mnt/c üzerinde çok yavaştı)."""
    cache = _ctx_cache_load()
    n = len(chunks)
    keys = [hashlib.md5(
                (CTX_MODEL + "||" + source + "||" + d.page_content).encode("utf-8")
            ).hexdigest() for d in chunks]

    ctxs = [cache.get(k) or "" for k in keys]           # cache'ten gelenler
    miss = [i for i, c in enumerate(ctxs) if not c]     # üretilmesi gerekenler
    tamam = n - len(miss)
    if on_progress and tamam:
        on_progress(tamam, n)

    if miss:
        with ThreadPoolExecutor(max_workers=CTX_PARALLEL) as ex:
            futs = {ex.submit(_uret_baglam, full_md, chunks[i].page_content): i
                    for i in miss}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    ctxs[i] = fut.result() or ""
                except Exception:
                    ctxs[i] = ""
                if ctxs[i]:
                    cache[keys[i]] = ctxs[i]
                tamam += 1
                if on_progress:
                    on_progress(tamam, n)
        _ctx_cache_save(cache)   # tek seferde diske yaz

    added = 0
    for doc, ctx in zip(chunks, ctxs):
        if ctx:
            doc.metadata["context"] = ctx
            doc.page_content = f"{ctx}\n\n{doc.page_content}"
            added += 1
    return chunks, added


def write_chunks_txt(chunks, output_dir, stem):
    """Chunk'ları gözle-kontrol için okunaklı bir .txt olarak yazar (MD'nin yanına)."""
    txt_path = os.path.join(output_dir, stem + "_chunks.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"KAYNAK: {stem}\nTOPLAM CHUNK: {len(chunks)}\n")
        f.write("=" * 78 + "\n")
        for d in chunks:
            no  = d.metadata.get("chunk_no", "?")
            ln  = len(d.page_content)
            tbl = "  [TABLO]" if d.metadata.get("has_table") else ""
            f.write(f"\n┌── CHUNK #{no}  ({ln} karakter){tbl} " + "─" * 36 + "\n")
            ctx  = d.metadata.get("context")
            body = d.page_content
            if ctx:
                f.write(f"│ [CONTEXTUAL BAĞLAM]\n│ {ctx}\n│\n")
                if body.startswith(ctx):
                    body = body[len(ctx):].lstrip("\n")
            else:
                f.write("│ [CONTEXTUAL BAĞLAM YOK !]\n│\n")
            for line in body.splitlines():
                f.write(f"│ {line}\n")
            f.write("└" + "─" * 70 + "\n")
    return txt_path


def kaynak_onizleme(doc, limit: int = 1500) -> str:
    """Referans panelindeki chunk önizlemesi. Tabloyu ortadan kesip bozmaz:
    kırpma gerekiyorsa tablo satırı ortasında değil, satır sınırında keser ve
    kalan kısmı özetler."""
    text = doc.page_content
    if len(text) <= limit:
        return text
    kesik = text[:limit]
    son_nl = kesik.rfind("\n")
    if son_nl > 0:
        kesik = kesik[:son_nl]
    kalan = len(text) - len(kesik)
    return f"{kesik}\n\n… *(önizleme kırpıldı — {kalan} karakter daha var; model tamamını görüyor)*"


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
        # TEK MODEL mimarisi: chunk/VLM her zaman otomatik vLLM gemma-4 kullanır.
        # Buradaki seçim yalnızca SOHBET modelini belirler: vLLM (varsayılan)
        # veya Gemini 3.0 Flash (bulut alternatifi). Başka seçenek yok.
        gemini_map = {
            "Gemini 3.0 Flash": "gemini-3-flash-preview",
        }
        model_options = [f"vLLM: {VLLM_MODEL_ID}"] + list(gemini_map.keys())

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
    """Seçili modele göre LangChain chat nesnesi kurar.

    TEK MODEL mimarisi:
      • "vLLM: ..."  → ChatOpenAI @ vLLM (gemma-4-12B-it). Sohbetin ANA modeli;
                        contextual ve VLM de aynı modeli (requests ile) kullanır.
      • "Gemini ..." → ChatGoogleGenerativeAI (bulut, opsiyonel).
    """
    sel = st.session_state.llm_option or ""
    temp = st.session_state.temperature

    if "vLLM" in sel:
        model_id = sel.split(": ", 1)[1]
        return ChatOpenAI(
            model=model_id,
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            temperature=temp,
            max_tokens=VLLM_MAX_TOKENS,
            streaming=True,
        )
    elif "Gemini" in sel:
        key = st.session_state.api_key
        if not key:
            st.error("⚠️ Google API Key girilmedi. Sol menüden API anahtarınızı girin.")
            return None
        gemini_map = {
            "Gemini 3.0 Flash": "gemini-3-flash-preview",
        }
        model_id = gemini_map.get(sel, "gemini-3-flash-preview")
        return ChatGoogleGenerativeAI(model=model_id, google_api_key=key, temperature=temp)

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
        return ("Model belleği doldu veya sunucu hata verdi. vLLM loglarını kontrol "
                "edin; gerekirse `--max-model-len` düşürülebilir ya da bulut modeli "
                "olarak Gemini seçilebilir.")
    if "does not support tools" in msg:
        return ("Seçili model araç çağırmayı (tool-calling) desteklemiyor. "
                "Lütfen Gemini'yi seçin veya tool destekli bir model kullanın.")
    if "connection" in msg or "refused" in msg or "max retries" in msg:
        return ("Model servisine bağlanılamadı. vLLM sunucusunun (:8000) çalıştığından "
                "veya Gemini API anahtarının doğru olduğundan emin olun.")
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
                        st.markdown(kaynak_onizleme(doc))
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
        # Gemini tool-calling destekler. vLLM (gemma-4) bu akışta tool-calling'e
        # sokulmaz → metin tabanlı sınıflandırma (router) ile yönlendiririz.
        # supports_tools yalnızca Gemini seçiliyken True'dur.
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
                context_str = "\n\n".join(
                    f"[KAYNAK {i+1}] ({d.metadata.get('source', '?')})\n{d.page_content}"
                    for i, d in enumerate(retrieved_docs)
                )
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
4. ZORUNLU: Cevabının EN SON satırına, tek başına şu formatta bir satır ekle:
   KAYNAKLAR: <cevabında gerçekten bilgi kullandığın kaynak numaraları, virgülle>
   Örnek: KAYNAKLAR: 1,3
   Hiçbir kaynaktan bilgi kullanmadıysan (cevap belgelerde yoksa): KAYNAKLAR: YOK

BELGELER:
{context_str}
"""
                rag_msgs = [SystemMessage(content=rag_prompt)] + history[:-1] + [HumanMessage(content=prompt)]
                chat_placeholder = st.empty()
                final_response = ""
                llm_hata = False

                # RAG Modu Akıllı Metin Akışı
                try:
                    for chunk in llm.stream(rag_msgs):
                        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                        final_response += content
                        gosterim = re.split(r"\n?\s*KAYNAKLAR\s*:", final_response)[0]
                        chat_placeholder.markdown(
                            f'<div class="msg-ai"><div class="bubble-ai">{gosterim}</div></div>',
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    llm_hata = True
                    final_response = _friendly_llm_error(e)
                    chat_placeholder.markdown(
                        f'<div class="msg-ai"><div class="access-deny-strip">⚠️ {final_response}</div></div>',
                        unsafe_allow_html=True,
                    )

                # ── Kaynak ayıklama ──
                # Model, cevabın sonunda hangi kaynakları GERÇEKTEN kullandığını
                # "KAYNAKLAR: 1,3" / "KAYNAKLAR: YOK" satırıyla bildirir. Bu satır
                # parse edilir, cevaptan silinir ve SADECE kullanılan kaynaklar
                # listelenir. Model satırı yazmadıysa yedek: "bilgi yok" kalıpları
                # taranır; onlar da yoksa geriye dönük uyumluluk için tümü gösterilir.
                m_k = re.search(r"KAYNAKLAR\s*:\s*(.+?)\s*$",
                                final_response, re.IGNORECASE | re.MULTILINE)
                temiz_cevap = re.sub(r"\n?\s*KAYNAKLAR\s*:.*$", "",
                                     final_response, flags=re.IGNORECASE | re.DOTALL).strip()
                kullanilan = []
                if llm_hata:
                    kullanilan = []
                elif m_k:
                    bildirim = m_k.group(1).strip().upper()
                    if "YOK" not in bildirim:
                        nums = [int(x) for x in re.findall(r"\d+", bildirim)]
                        kullanilan = [retrieved_docs[n - 1] for n in dict.fromkeys(nums)
                                      if 1 <= n <= len(retrieved_docs)]
                else:
                    yok_kaliplari = ("bulunmuyor", "bilgim yok", "bilgi yok", "bilgi bulunmamaktadır",
                                     "yer almamaktadır", "bulunamadı", "mevcut değil", "erişilemiyor")
                    if not any(k in final_response.lower() for k in yok_kaliplari):
                        kullanilan = retrieved_docs

                if temiz_cevap and not llm_hata:
                    final_response = temiz_cevap
                    chat_placeholder.markdown(
                        f'<div class="msg-ai"><div class="bubble-ai">{final_response}</div></div>',
                        unsafe_allow_html=True,
                    )
                retrieved_docs = kullanilan

                if retrieved_docs:
                    with st.expander(f"🔍 Referans Kaynaklar ({len(retrieved_docs)})"):
                        for i, doc in enumerate(retrieved_docs):
                            score = doc.metadata.get("score", 0.0)
                            st.markdown(f"**#{i+1}** &nbsp; 📄 `{doc.metadata.get('source')}` &nbsp; 📊 Skor: `{score:.4f}`")
                            st.markdown(kaynak_onizleme(doc))
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
                # vLLM (gemma-4): taze sohbet yanıtı üret
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
                            try:
                                ensure_ctx_model()   # gemma/vlm kurulu değilse sessiz boş bağlam yerine burada dur
                            except RuntimeError as e:
                                s.update(label=f"❌ {e}", state="error")
                                st.stop()
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=os.path.splitext(f_name)[1]
                            ) as tmp:
                                tmp.write(bytes_data)
                                tmp_path = tmp.name
                            prog = st.progress(0.0, text="Başlatılıyor...")
                            s.write("Koleksiyon hazırlanıyor...")
                            init_collection()
                            os.makedirs(OUTPUT_DIR, exist_ok=True)
                            prog.progress(0.10, text="Belge işleniyor (Docling + VLM görsel analizi)...")
                            s.write("Belge işleniyor (Docling + VLM görsel analizi)...")
                            full_md = belge_to_md(tmp_path, OUTPUT_DIR, orijinal_ad=f_name)
                            prog.progress(0.50, text="Chunklanıyor (başlık-tabanlı + tablo-atomik)...")
                            s.write("Chunklanıyor (başlık-tabanlı + tablo-atomik)...")
                            chunks = chunk_md(
                                full_md, f_name,
                                st.session_state.chunk_size,
                                st.session_state.chunk_overlap,
                                audience_dict,
                            )
                            prog.progress(0.55, text=f"Contextual bağlam üretiliyor (0/{len(chunks)})...")
                            s.write(f"{len(chunks)} chunk için contextual bağlam üretiliyor ({CTX_MODEL})...")
                            chunks, ctx_added = add_contextual(
                                chunks, full_md, f_name,
                                on_progress=lambda i, n: prog.progress(
                                    0.55 + 0.35 * (i / n),
                                    text=f"Contextual bağlam üretiliyor ({i}/{n})...",
                                ),
                            )
                            if ctx_added < len(chunks):
                                s.write(f"⚠️ {len(chunks) - ctx_added} chunk bağlamsız kaldı (vLLM loglarına bak).")
                            os.unlink(tmp_path)
                            if chunks:
                                prog.progress(0.92, text=f"{len(chunks)} chunk Qdrant'a yükleniyor...")
                                s.write(f"{len(chunks)} chunk Qdrant'a yükleniyor...")
                                delete_by_source(f_name)   # eskiler ancak yeni chunk'lar HAZIRKEN silinir
                                add_documents_to_qdrant(chunks, file_hash=curr_md5)
                                # MD'nin yanına chunk kontrol dosyasını da yaz
                                stem_ = os.path.splitext(f_name)[0]
                                chunks_txt = write_chunks_txt(chunks, OUTPUT_DIR, stem_)
                                s.write(f"Chunk kontrol dosyası yazıldı: {chunks_txt}")
                                prog.progress(1.0, text="Tamamlandı ✓")
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