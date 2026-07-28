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
import os, time
import requests
import jwt as pyjwt   # PyJWT — test JWT imzalamak için (bkz. API_JWT ayarları)

import streamlit as st
from abac import (
    AudiencePolicy, UserContext,
    has_access, build_policy_from_ui, parse_ids, empty_rule_data, FIELD_LABELS,
)
from tenancy import ConventionTenantRegistry
import base64

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Bilimp AI Asistan",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

# ── Heavy imports (cached via @st.cache_resource) ────────────────────────────
from qdrant_client import QdrantClient

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

# ── vLLM (OpenAI-uyumlu) — TEK MODEL, her iş burada ──
# Sohbet, contextual bağlam üretimi ve görsel (VLM) açıklaması aynı modelden:
# google/gemma-4-12B-it (Unified, encoder-free multimodal).
# vLLM auth istemiyor; api_key sadece istemcinin zorunlu tuttuğu placeholder.
VLLM_BASE_URL   = f"http://{_HOST_IP}:8000/v1"
VLLM_API_KEY    = "EMPTY"
VLLM_MODEL_ID   = "google/gemma-4-12B-it"
VLLM_MAX_TOKENS = 2048

# ── Bilimp API (auth.py + api.py, JWT ile korunan gerçek sistem) ──
# (v2.1) Sohbet artık burada (Streamlit içinde) Qdrant/LLM'e DOĞRUDAN
# girmez — auth.py'nin JWT doğrulamasından, tenancy.py'nin fiziksel tenant
# izolasyonundan ve rag_service.py'nin ABAC filtresinden GERÇEKTEN geçmesi
# için api.py'nin POST /api/v1/query ucunu HTTP üzerinden çağırır. Streamlit
# artık ince bir istemci; tek doğruluk kaynağı api.py'dir (bkz. POC branch).
# api.py 8000'i vLLM kullandığı için varsayılan olarak 8001'de çalıştırılmalı
# (örn. `uvicorn api:app --port 8001`).
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{_HOST_IP}:8001")

# Bu Streamlit sayfası "Bilimp"in kendisi DEĞİLDİR — gerçek entegrasyonda JWT'yi
# Bilimp'in kendi backend'i üretir. Burada SADECE demo/test amaçlı, Kimlik Girişi
# sayfasında seçilen özniteliklerden yerel bir özel anahtarla (private_key.pem —
# bkz. generate_test_keypair.py) kendi test token'ımızı imzalıyoruz; auth.py
# yalnızca GENEL anahtarı (.env: JWT_PUBLIC_KEY) bilir, bu imzalama gerçek
# ortamda ASLA yapılmaz.
API_JWT_ISSUER = os.getenv("JWT_ISSUER", "bilimp-teracity")
API_JWT_AUDIENCE_QUERY = os.getenv("JWT_AUDIENCE_QUERY", "tubitak1505-query")
API_JWT_AUDIENCE_ADMIN = os.getenv("JWT_AUDIENCE_ADMIN", "tubitak1505-audience-admin")
API_JWT_PRIVATE_KEY_PATH = os.getenv("API_JWT_PRIVATE_KEY_PATH", "private_key.pem")

# api.py'nin GERÇEKTEN yazdığı tenant koleksiyon adını hesaplamak için — bkz.
# tenancy.py. Belge Yönetimi listeleme paneli (salt-okunur) bunu kullanır.
tenant_registry = ConventionTenantRegistry()

# (v2.1) Docling+VLM+contextual ingestion ayarları BURADAN KALDIRILDI — bu
# pipeline artık vllm_ingestion.py'de (api.py'nin document_ingestion_service.py
# üzerinden çağırdığı) yaşıyor. Streamlit artık belge yüklerken bunu
# ÇALIŞTIRMAZ, api.py'yi HTTP ile çağırır (bkz. page_documents).

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


@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, check_compatibility=False)


# (v2.1) init_collection / add_documents_to_qdrant / delete_by_source /
# delete_document_globally / get_vllm_models / sync_registry (+ load_registry /
# save_registry / belge_kayitlari.json kayıt defteri) KALDIRILDI — belge
# yazma/silme artık api.py'yi (auth + tenancy + document_ingestion_service.py)
# HTTP ile çağırır (bkz. page_documents, _documents_api). get_dense_embeddings /
# get_sparse_embeddings de yalnızca bu kaldırılan doğrudan-Qdrant-yazma yolunda
# kullanılıyordu; embedding artık api.py'nin kendi lifespan'inde kurulur.


# (v2.1) Docling+VLM+contextual ingestion pipeline'i (belge_to_md, chunk_md,
# add_contextual, _call_vlm, ensure_ctx_model + tum yardimcilari) BURADAN
# vllm_ingestion.py'ye tasindi; api.py'nin document_ingestion_service.py'si
# bunu HTTP uzerinden (auth+tenancy korumali /api/v1/documents uclariyla)
# cagirir. Streamlit artik bu pipeline'i KENDI icinde CALISTIRMAZ (bkz.
# page_documents, _documents_api) - iki ayri kopya tutmak (DRY ihlali,
# birbirinden sapma riski) yerine tek dogruluk kaynagi vllm_ingestion.py'dir.
# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "page":           "chat",
        "messages":       [],
        "audience_rules": [empty_rule_data()],
        # UserContext fields
        # (v2.0) musteri_id: fiziksel tenant sınırı, UserContext'te ZORUNLU
        # alan (bkz. abac.py / tenancy.py). 0 = henüz girilmedi; bu durumda
        # sorgu var olmayan bir koleksiyona gider ve deny-by-default olarak
        # boş sonuç döner (bkz. rag_service.retrieve_authorized_docs) — hata
        # fırlatmaz ama hiçbir belge de bulunmaz.
        "uc_musteri":     0,
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
        # (v2.0) musteri_id ZORUNLU — Optional DEĞİL, bu yüzden 'or None'
        # kullanılmaz (None geçilirse Pydantic doğrulaması patlar).
        musteri_id      = st.session_state.uc_musteri,
        # (v2.0) sirket_ids artık ÇOĞUL/kesişim modelinde — grup_ids ile
        # aynı desen. Demo formu tek bir şirket ID'si topluyor, bunu tek
        # elemanlı listeye çeviriyoruz.
        sirket_ids      = [st.session_state.uc_sirket] if st.session_state.uc_sirket else [],
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
    if uc.musteri_id:
        parts.append(f'<span class="id-label">Müşteri</span><span class="id-chip">{uc.musteri_id}</span>')
    if uc.sirket_ids:
        sirk = " ".join(f'<span class="id-chip">{s}</span>' for s in uc.sirket_ids)
        parts.append(f'<span class="id-label">Şirket</span>{sirk}')
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
    if uc.musteri_id:  badges.append(f'<span class="profile-badge">Müşteri {uc.musteri_id}</span>')
    for s in uc.sirket_ids: badges.append(f'<span class="profile-badge">Şirket {s}</span>')
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
        st.markdown('<h3>🧠 Model</h3>', unsafe_allow_html=True)
        # (v2.1) Sohbet artık Streamlit içinde model seçtirmez — api.py'ye
        # HTTP ile bağlanır (bkz. page_chat/_call_query_api) ve hangi LLM'in
        # kullanılacağı SUNUCU tarafında LLM_PROVIDER ortam değişkeniyle
        # belirlenir (bkz. api.py::_build_llm). Burada bir seçim sunmak
        # kullanıcıyı yanıltırdı — seçilen şey isteğe hiç yansımıyordu.
        st.caption(
            f"Sohbet, Bilimp API'sinin ({API_BASE_URL}) sunucu tarafı "
            "LLM_PROVIDER ayarını kullanır (varsayılan: vLLM)."
        )

        # (v2.1) Chunk boyutu/örtüşmesi artık istemciden ayarlanamaz — belge
        # yönetimi api.py'ye taşındığından, chunking document_ingestion_
        # service.py'de SUNUCU tarafı sabitleriyle (_DEFAULT_CHUNK_SIZE/
        # _DEFAULT_CHUNK_OVERLAP) yapılır; musteri_id gibi diğer sunucu-taraflı
        # kararlarla AYNI ilkeyle, istek parametresi olarak sunulmaz.

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
        "musteri": 501, "sirket": 14, "sube": 0, "mudurlu": 25, "birim": 0, "bina": 0,
        "pozisyon": 1, "ptype": 0, "kullanici": 0, "grup": "1",
    },
    "c.erdem (Analiz Destek - Özel Yetkili)": {
        "musteri": 501, "sirket": 14, "sube": 0, "mudurlu": 55, "birim": 0, "bina": 0,
        "pozisyon": 0, "ptype": 0, "kullanici": 590, "grup": "",
    },
    "Yazılım Süreç Yöneticisi (Yazılım + Yönetim)": {
        "musteri": 501, "sirket": 14, "sube": 0, "mudurlu": 13, "birim": 0, "bina": 0,
        "pozisyon": 0, "ptype": 0, "kullanici": 0, "grup": "101",
    },
    "Özlüce Kampüsü Çalışanı (Memur)": {
        "musteri": 501, "sirket": 14, "sube": 0, "mudurlu": 0, "birim": 0, "bina": 16,
        "pozisyon": 0, "ptype": 1, "kullanici": 0, "grup": "",
    },
    MANUAL_PROFILE: {
        "musteri": 0, "sirket": 0, "sube": 0, "mudurlu": 0, "birim": 0, "bina": 0,
        "pozisyon": 0, "ptype": 0, "kullanici": 0, "grup": "",
    },
}


# Form widget anahtarları (f_*) ↔ kalıcı kaynak anahtarları (uc_*) eşlemesi.
# f_* widget'lara bağlıdır ve sayfadan ayrılınca purge olur; uc_* ise düz
# session_state anahtarıdır ve sayfa değişse de kalıcıdır (kimlik korunur).
_AUTH_FIELDS = [
    ("f_musteri", "uc_musteri"),
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
    st.session_state.f_musteri   = p["musteri"]
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

        st.number_input(
            "Müşteri ID (zorunlu — fiziksel tenant sınırı)",
            0, 99999, key="f_musteri",
            help="Bilimp müşteri (hesap) kimliği. Sorgu yalnızca bu müşterinin "
                 "koleksiyonunda çalışır; 0 bırakılırsa hiçbir belgeye erişilemez.",
        )
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
            "musteri_id":      uc.musteri_id,
            "sirket_ids":      uc.sirket_ids,
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
def _load_demo_private_key() -> str:
    """
    DEMO/TEST AMAÇLI: GERÇEK Bilimp entegrasyonunda bu imzalama adımı YOKTUR
    — token'ı Bilimp'in kendi backend'i üretir ve istemciye verir; auth.py
    yalnızca GENEL anahtarı (.env: JWT_PUBLIC_KEY) bilir, ÖZEL anahtarı asla
    görmez (bkz. auth.py başlığı, RS256 notu). Bu Streamlit sayfası "Bilimp"
    DEĞİL — bir iç test/demo aracı; api.py'yi GERÇEKTEN JWT'li çağırabilmek
    için generate_test_keypair.py'nin ürettiği private_key.pem ile kendi test
    token'larımızı burada imzalıyoruz (generate_test_token.py ile AYNI desen).
    """
    if not os.path.exists(API_JWT_PRIVATE_KEY_PATH):
        raise RuntimeError(
            f"'{API_JWT_PRIVATE_KEY_PATH}' bulunamadı. Önce çalıştırın: "
            "`python generate_test_keypair.py` — ardından public_key.pem içeriğini "
            ".env'deki JWT_PUBLIC_KEY değerine koyup api.py'yi yeniden başlatın."
        )
    with open(API_JWT_PRIVATE_KEY_PATH) as f:
        return f.read()


def _sign_demo_user_jwt(uc: UserContext) -> str:
    """Kullanıcı token'ı — POST /api/v1/query için (aud=JWT_AUDIENCE_QUERY)."""
    private_key = _load_demo_private_key()
    now = int(time.time())
    payload = {
        "iss": API_JWT_ISSUER,
        "aud": API_JWT_AUDIENCE_QUERY,
        "sub": str(uc.kullanici_id) if uc.kullanici_id is not None else "demo-kullanici",
        "iat": now,
        "exp": now + 3600,
        "user_context": uc.model_dump(),
    }
    return pyjwt.encode(payload, private_key, algorithm="RS256")


def _sign_demo_service_jwt(musteri_id: int) -> str:
    """Servis token'ı — doküman/hedef kitle yönetim uçları için
    (aud=JWT_AUDIENCE_ADMIN). musteri_id, Kimlik Girişi sayfasında seçilen
    tenant'tır — bu demoda 'hangi müşterinin belgelerini yönettiğiniz' ile
    'hangi müşterinin kullanıcısı gibi sorgu attığınız' AYNI seçimdir."""
    private_key = _load_demo_private_key()
    now = int(time.time())
    payload = {
        "iss": API_JWT_ISSUER,
        "aud": API_JWT_AUDIENCE_ADMIN,
        "sub": "bilimp-backend",
        "musteri_id": musteri_id,
        "iat": now,
        "exp": now + 3600,
    }
    return pyjwt.encode(payload, private_key, algorithm="RS256")


def _friendly_api_error(e: Exception) -> str:
    """api.py çağrısı hatalarını kullanıcı dostu Türkçe mesaja çevirir."""
    msg = str(e).lower()
    if "connection" in msg or "refused" in msg or "max retries" in msg or "timeout" in msg:
        return (f"Bilimp API'sine ({API_BASE_URL}) bağlanılamadı. Servisin çalıştığından "
                "emin olun (örn. `uvicorn api:app --port 8001`).")
    return f"API isteği başarısız: {e}"


def _call_query_api(token: str, soru: str, gecmis: list[dict]) -> dict:
    """POST /api/v1/query — auth.py + tenancy.py + rag_service.py'nin GERÇEK
    zincirinden geçer. Streaming YOKTUR (API kontratı tam yanıt döner; bkz.
    rag_service.answer_with_context docstring'i)."""
    r = requests.post(
        f"{API_BASE_URL}/api/v1/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"soru": soru, "gecmis_mesajlar": gecmis},
        timeout=180,
    )
    if r.status_code != 200:
        try:
            detay = r.json().get("mesaj", r.text)
        except Exception:
            detay = r.text
        raise RuntimeError(f"API {r.status_code}: {detay}")
    return r.json()


def _kaynak_satiri(kaynak: dict, i: int) -> str:
    """API'nin QuerySource şeması: dokuman_id/dosya_adi/sayfa/versiyon/skor —
    chunk İÇERİĞİ dönmez (bkz. api_schemas.py). Bu kasıtlı bir API sözleşmesi
    kararıdır (Bilimp'e ham chunk metni değil yalnızca kaynak künyesi verilir);
    bu yüzden eski kaynak_onizleme() içerik önizlemesi burada YOKTUR."""
    parca = (
        f"**#{i+1}** &nbsp; 📄 `{kaynak.get('dosya_adi') or kaynak.get('dokuman_id')}` &nbsp; "
        f"📊 Skor: `{kaynak.get('skor', 0.0):.4f}`"
    )
    if kaynak.get("sayfa"):
        parca += f" &nbsp; 📖 Sayfa: `{kaynak['sayfa']}`"
    return parca


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
            if m.get("sources"):
                with st.expander(f"🔍 Referans Kaynaklar ({len(m['sources'])})"):
                    for i, kaynak in enumerate(m["sources"]):
                        st.markdown(_kaynak_satiri(kaynak, i))
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

        # api_schemas.QueryRequest.gecmis_mesajlar: kronolojik, {rol, icerik},
        # en fazla 20 mesaj. Az önce eklenen kullanıcı mesajı (bu turun kendisi)
        # HARİÇ tutulur — o zaten ayrı 'soru' alanında gönderilir.
        gecmis = []
        for msg in st.session_state.messages[-21:-1]:
            rol = "kullanici" if msg["role"] == "user" else "asistan"
            icerik = (msg.get("content") or "").strip()
            if icerik:
                gecmis.append({"rol": rol, "icerik": icerik[:8000]})

        final_response = ""
        retrieved_docs: list = []

        with st.spinner("Bilimp API'sine soruluyor (auth + tenancy + ABAC + vLLM)..."):
            try:
                token = _sign_demo_user_jwt(uc)
                sonuc = _call_query_api(token, prompt, gecmis)
                durum = sonuc.get("durum")
                final_response = sonuc.get("yanit", "")
                retrieved_docs = sonuc.get("kaynaklar", []) or []
                basarisiz = False
            except Exception as e:
                durum = None
                final_response = _friendly_api_error(e)
                basarisiz = True

        if basarisiz or durum != "basarili":
            st.markdown(
                f'<div class="msg-ai"><div class="access-deny-strip">⛔ {final_response}</div></div>',
                unsafe_allow_html=True,
            )
            retrieved_docs = []
        else:
            st.markdown(
                '<div class="lang-strip">📚 Bilimp API üzerinden yanıtlandı (JWT doğrulandı)</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="msg-ai"><div class="bubble-ai">{final_response}</div></div>',
                unsafe_allow_html=True,
            )
            if retrieved_docs:
                with st.expander(f"🔍 Referans Kaynaklar ({len(retrieved_docs)})"):
                    for i, kaynak in enumerate(retrieved_docs):
                        st.markdown(_kaynak_satiri(kaynak, i))
                        if i < len(retrieved_docs) - 1:
                            st.divider()

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


def _documents_api(method: str, path: str, token: str, json_body: dict | None = None):
    """
    api.py'nin doküman yönetim uçlarına (/api/v1/documents...) GERÇEK HTTP
    isteği atar — servis token'ıyla, auth.verify_service_token'dan gerçekten
    geçer. Bu, eski (v2.0) davranışın Qdrant'a DOĞRUDAN yazmasının yerini alır.

    Dönüş: (basarili: bool, http_status: int, govde: dict|str)
    """
    try:
        r = requests.request(
            method, f"{API_BASE_URL}/api/v1/documents{path}",
            headers={"Authorization": f"Bearer {token}"},
            json=json_body, timeout=180,
        )
    except Exception as e:
        return False, 0, _friendly_api_error(e)
    try:
        govde = r.json()
    except Exception:
        govde = r.text
    return r.status_code in (200, 201), r.status_code, govde


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
            st.success(f"✅ **{f_name}** yüklenmeye hazır ({len(bytes_data) / 1024:.1f} KB).")

        st.divider()
        _render_abac_builder()
        st.divider()

        if up_file:
            uc = get_user_context()
            if not uc.musteri_id:
                st.warning("⚠️ Önce 'Kimlik Girişi' sayfasından bir Müşteri ID girin — "
                           "servis token'ı hangi tenant'ta işlem yapılacağını bundan alır.")

            col_btn, _ = st.columns([2, 1])
            with col_btn:
                if st.button("🚀 api.py Üzerinden Yayınla", type="primary",
                             use_container_width=True, disabled=not uc.musteri_id):
                    audience_policy = build_policy_from_ui(st.session_state.audience_rules)
                    if audience_policy.is_empty():
                        st.error("❌ En az bir geçerli erişim kuralı tanımlanmalıdır. Boş kayıt sisteme kabul edilmez.")
                    else:
                        # (v2.1) Belge yönetimi artık Qdrant'a DOĞRUDAN yazmaz —
                        # auth.py'nin servis token doğrulamasından, tenancy.py'nin
                        # fiziksel tenant izolasyonundan ve document_ingestion_
                        # service.py'nin GERÇEK Docling+VLM+contextual pipeline'ından
                        # (vllm_ingestion.py) geçmesi için api.py'yi HTTP ile çağırır.
                        kullanici_id = uc.kullanici_id or 1
                        audience_dict = audience_policy.model_dump()
                        icerik_b64 = base64.b64encode(bytes_data).decode()

                        with st.status("🔄 api.py'ye gönderiliyor...", expanded=True) as s:
                            try:
                                token = _sign_demo_service_jwt(uc.musteri_id)
                            except RuntimeError as e:
                                s.update(label=f"❌ {e}", state="error")
                                st.stop()

                            s.write(f"'{f_name}' bu tenant'ta mevcut mu kontrol ediliyor (GET /audience)...")
                            var_mi, durum_kodu, mevcut = _documents_api(
                                "GET", f"/{f_name}/audience", token
                            )

                            if not var_mi and durum_kodu == 404:
                                s.write("Yeni doküman oluşturuluyor — Docling dönüşümü + VLM görsel "
                                        "açıklama + contextual bağlam (vLLM üzerinden, POST /documents)...")
                                basarili, sk, govde = _documents_api(
                                    "POST", "", token,
                                    {
                                        "dokuman_id": f_name,
                                        "dosya_icerigi_base64": icerik_b64,
                                        "audience_policy": audience_dict,
                                        "yukleyen_kullanici_id": kullanici_id,
                                    },
                                )
                            elif var_mi:
                                icerik_versiyonu = mevcut.get("icerik_versiyonu") or 1
                                audience_versiyonu = mevcut.get("audience_versiyon", 1)
                                s.write(f"Mevcut belge güncelleniyor (PUT /content, "
                                        f"beklenen_versiyon={icerik_versiyonu})...")
                                basarili, sk, govde = _documents_api(
                                    "PUT", f"/{f_name}/content", token,
                                    {
                                        "dosya_icerigi_base64": icerik_b64,
                                        "beklenen_versiyon": icerik_versiyonu,
                                        "degistiren_kullanici_id": kullanici_id,
                                    },
                                )
                                if basarili:
                                    s.write("Hedef kitle politikası güncelleniyor (PUT /audience)...")
                                    aud_ok, aud_sk, aud_govde = _documents_api(
                                        "PUT", f"/{f_name}/audience", token,
                                        {
                                            "audience_policy": audience_dict,
                                            "beklenen_audience_versiyon": audience_versiyonu,
                                            "degistiren_kullanici_id": kullanici_id,
                                        },
                                    )
                                    if not aud_ok:
                                        s.write(f"⚠️ İçerik güncellendi ama hedef kitle güncellenemedi "
                                                f"({aud_sk}): {aud_govde}")
                            else:
                                basarili, sk, govde = False, durum_kodu, mevcut

                            if basarili:
                                s.update(label=f"✅ '{f_name}' api.py üzerinden yayınlandı! "
                                              f"({govde})", state="complete")
                                st.toast("Belge api.py üzerinden sisteme entegre edildi!", icon="🎉")
                                time.sleep(0.8)
                                st.rerun()
                            else:
                                s.update(label=f"❌ API hatası ({sk}): {govde}", state="error")

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

        # (v2.1) api.py'nin doküman uçlarında "tüm belgeleri listele" karşılığı
        # bir uç YOKTUR (API Kontrat Dokümanı v0.1 kasıtlı olarak yalnızca
        # tekil doküman/audience-compliance-report uçları sunar — Bilimp
        # kendi doküman listesini kendi tutar). Bu yüzden bu panel Qdrant'ı
        # DOĞRUDAN (salt-okunur) tarar; bu, hiçbir yazma işlemi yapmadığından
        # auth/tenancy güvenlik sınırını ihlal etmez, yalnızca API'nin
        # sunmadığı bir kolaylık görünümüdür. Doğru tenant koleksiyonu (API'nin
        # GERÇEKTEN yazdığı yer) tenancy.py ile AYNI kuralla çözümlenir.
        uc = get_user_context()
        client = get_qdrant_client()
        doc_map: dict[str, dict] = {}

        if uc.musteri_id:
            collection = tenant_registry.collection_name(uc.musteri_id)
            if client.collection_exists(collection):
                scroll, _ = client.scroll(collection_name=collection, limit=2000, with_payload=True)
                for pt in scroll:
                    meta = pt.payload.get("metadata", {})
                    src = meta.get("source", "")
                    if src and src not in doc_map:
                        doc_map[src] = {
                            "audience": meta.get("audience", {}),
                            "file_type": meta.get("file_type", "?"),
                            "versiyon": meta.get("versiyon", 1),
                        }

        if not uc.musteri_id:
            st.info("Belgeleri görmek için önce 'Kimlik Girişi' sayfasından bir Müşteri ID girin.")
        elif not doc_map:
            st.info("Bu müşteride henüz belge yüklenmemiş.")
        else:
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
                    if st.button("🗑", key=f"del_{fname}", help="Belgeyi api.py üzerinden sil"):
                        service_token = _sign_demo_service_jwt(uc.musteri_id)
                        kullanici_id = uc.kullanici_id or 1
                        basarili, sk, govde = _documents_api(
                            "DELETE", f"/{fname}", service_token,
                            {
                                "beklenen_versiyon": info.get("versiyon", 1),
                                "degistiren_kullanici_id": kullanici_id,
                            },
                        )
                        if basarili:
                            st.toast(f"'{fname}' api.py üzerinden silindi.", icon="🗑")
                            st.rerun()
                        else:
                            st.error(f"❌ Silme başarısız ({sk}): {govde}")
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
