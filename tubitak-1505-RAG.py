import os
import time
import json
import hashlib
import tempfile
import requests
import torch
import re
from uuid import uuid4

import streamlit as st
from streamlit_lottie import st_lottie

# --- GEREKLİ IMPORTLAR ---
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from language_utils import choose_answer_language, build_language_policy_prompt, get_language_label

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Bilimp AI Asistan", layout="wide", page_icon="🤖")


# --- ANİMASYON YÜKLEME ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except:
        return None


# --- YARDIMCI FONKSİYON: METİN AKIŞI SİMÜLASYONU ---
def stream_text_generator(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)


# --- GÖRSEL YÜKLEME EKRANI ---
if "app_loaded" not in st.session_state:
    loader_placeholder = st.empty()
    with loader_placeholder.container():
        st.markdown(
            """<style>.stApp {background-color: #0e1117;} .glowing-text {font-family: 'Source Code Pro', monospace; color: #00fbff; text-align: center; font-size: 2em; font-weight: bold; text-shadow: 0 0 10px #00fbff; animation: pulse 1.5s infinite;} @keyframes pulse { from {opacity: 0.8;} to {opacity: 1;} }</style>""",
            unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            lottie_json = load_lottieurl("https://lottie.host/file/9860f43c-6232-4665-ba4f-557c669299b6.json")
            if lottie_json: st_lottie(lottie_json, height=250, key="loader", speed=1.5)

        status_text_placeholder = st.empty()
        loading_steps = ["🧠 Nöral Ağlar Yükleniyor...", "⚡ GPU Hızlandırma Aktif...",
                         "🛠️ Streaming (Akış) Modülü Başlatılıyor...", "🚀 Lütfen Bekleyiniz Sistem Hazırlanıyor..."]
        for step in loading_steps:
            status_text_placeholder.markdown(f'<p class="glowing-text">{step}</p>', unsafe_allow_html=True)
            time.sleep(0.5)

        # --- IMPORTLAR ---
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as rest_models
        from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, Filter, FieldCondition, \
            MatchValue, MatchAny
        from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_ollama import ChatOllama
        from langchain_core.output_parsers import StrOutputParser
        import pymupdf4llm
        from markitdown import MarkItDown
        from pptx import Presentation
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

    loader_placeholder.empty()
    st.session_state["app_loaded"] = True
else:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest_models
    from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, Filter, FieldCondition, \
        MatchValue, MatchAny
    from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser
    import pymupdf4llm
    from markitdown import MarkItDown
    from pptx import Presentation
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

# ==============================================================================
# AYARLAR
# ==============================================================================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "Tubitak_Dokumanlar_Hybrid"
EMBEDDING_MODEL_NAME = "ytu-ce-cosmos/turkish-e5-large"
REGISTRY_FILE = "belge_kayitlari.json"


# ==============================================================================
# YARDIMCI FONKSİYONLAR
# ==============================================================================
def calculate_md5(file_bytes):
    hash_md5 = hashlib.md5()
    hash_md5.update(file_bytes)
    return hash_md5.hexdigest()


def load_registry():
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_registry(registry):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=4)


# ==============================================================================
# CHUNKLAMA VE PARSE İŞLEMLERİ
# ==============================================================================
def etiketleri_generic_duzelt(text):
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") or (stripped.startswith("**") and stripped.endswith("**")):
            new_lines.append(line)
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)


def process_pptx_native(file_path, source_name, permission):
    prs = Presentation(file_path)
    slides_chunks = []
    for i, slide in enumerate(prs.slides):
        content = []
        if slide.shapes.title and slide.shapes.title.text:
            content.append(f"# {slide.shapes.title.text.strip()}")
        for shape in slide.shapes:
            if hasattr(shape, "text_frame") and shape.text_frame:
                content.append(shape.text.strip())
        full = "\n\n".join(content)
        if full.strip():
            doc = Document(page_content=full, metadata={"source": source_name, "chunk_no": i + 1, "file_type": "pptx",
                                                        "permission": permission})
            slides_chunks.append(doc)
    return slides_chunks


def process_text_based(file_path, source_name, chunk_size, chunk_overlap, permission):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        # 1. Markdown Dönüşümü
        if ext == ".pdf":
            text = pymupdf4llm.to_markdown(file_path, write_images=False)
        else:
            md = MarkItDown()
            result = md.convert(file_path)
            text = result.text_content

        # Temizlik
        clean = etiketleri_generic_duzelt(text)

        # 2. Başlıklara Göre Bölme
        headers_to_split_on = [
            ("#", "Main"),
            ("##", "Sub"),
            ("###", "Sub2"),
            ("####", "Sub3")
        ]

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=True
        )
        md_docs = splitter.split_text(clean)

        # --- AKILLI BİRLEŞTİRME 2.0 (SAFE MERGE) ---
        merged_docs = []
        temp_doc = None

        for doc in md_docs:
            if not doc.page_content.strip():
                continue

            # Context (Bağlam) bilgisini hazırla
            header_path = " > ".join([doc.metadata.get(h[1]) for h in headers_to_split_on if doc.metadata.get(h[1])])
            if header_path:
                doc.page_content = f"**BAĞLAM:** {header_path}\n\n{doc.page_content}"

            # Eğer elimizde bekleyen "yetim" bir parça varsa:
            if temp_doc:
                if len(doc.page_content) < 100 and "|" not in doc.page_content:
                    merged_docs.append(temp_doc)
                    temp_doc = doc
                else:
                    new_content = f"{temp_doc.page_content}\n\n{doc.page_content}"
                    doc.page_content = new_content
                    merged_docs.append(doc)
                    temp_doc = None

            else:
                if len(doc.page_content) < 250 and "|" not in doc.page_content:
                    temp_doc = doc
                else:
                    merged_docs.append(doc)

        if temp_doc:
            merged_docs.append(temp_doc)

        # 3. Recursive Splitter (Çok büyükleri bölmek için)
        rec_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        final_docs = []
        for doc in merged_docs:
            doc.metadata.update({
                "source": source_name,
                "file_type": ext.replace(".", ""),
                "permission": permission
            })
            chunks = rec_splitter.split_documents([doc])
            final_docs.extend(chunks)

        return final_docs

    except Exception as e:
        st.error(f"Hata: {e}")
        return []


# ==============================================================================
# QDRANT VE EMBEDDING MODELLERİ
# ==============================================================================
@st.cache_resource
def get_dense_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device},
                                 encode_kwargs={"normalize_embeddings": True})


@st.cache_resource
def get_sparse_embeddings():
    return FastEmbedSparse(model_name="Qdrant/bm25")


@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, check_compatibility=False)


def init_collection():
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"content": VectorParams(size=1024, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()}
        )


def add_documents_to_qdrant(documents, file_hash=None):
    """
    documents: belge listesi
    file_hash: dosyanın MD5 hash'i (metadata'ya eklemek için)
    """
    client = get_qdrant_client()
    dense_emb = get_dense_embeddings()
    sparse_emb = get_sparse_embeddings()

    # Hash'i metadata'ya ekle
    if file_hash:
        for doc in documents:
            doc.metadata["file_hash"] = file_hash

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=dense_emb,
        vector_name="content",
        sparse_embedding=sparse_emb,
        sparse_vector_name="sparse",
        retrieval_mode=RetrievalMode.HYBRID
    )
    ids = [str(uuid4()) for _ in documents]
    vector_store.add_documents(documents=documents, ids=ids)


def delete_by_source(source_name):
    client = get_qdrant_client()
    if client.collection_exists(COLLECTION_NAME):
        client.delete(collection_name=COLLECTION_NAME, points_selector=Filter(
            must=[FieldCondition(key="metadata.source", match=MatchValue(value=source_name))]))


def delete_document_globally(filename):
    delete_by_source(filename)
    reg = load_registry()
    if filename in reg:
        del reg[filename]
        save_registry(reg)


def get_allowed_permissions(role):
    hierarchy = {
        "public": ["public"],
        "user": ["public", "user"],
        "management": ["public", "user", "management"],
        "admin": ["public", "user", "management", "admin", "private"],
        "private": ["private"]
    }
    return hierarchy.get(role, ["public"])


def get_local_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=1)
        if response.status_code == 200:
            return [m["name"] for m in response.json().get("models", [])]
    except:
        return []
    return []


# ==============================================================================
# SENKRONIZASYON (App Başlaması Sırasında)
# ==============================================================================
def sync_registry_with_qdrant():
    """
    Qdrant'taki dökümanları baz alarak JSON'u güncelle.
    Qdrant'ta varsa ve JSON'da yoksa -> JSON'a ekle
    JSON'da varsa ve Qdrant'ta yoksa -> JSON'dan sil

    Hash bilgisini Qdrant'tan çıkar (eğer varsa)
    """
    client = get_qdrant_client()
    registry = load_registry()

    qdrant_files = {}  # {filename: {"permission": perm, "hash": hash}}

    if client.collection_exists(COLLECTION_NAME):
        scroll_result = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True
        )

        # Qdrant'tan benzersiz dosyaları topla (hash dahil)
        for point in scroll_result[0]:
            metadata = point.payload.get("metadata", {})
            source = metadata.get("source", "")
            perm = metadata.get("permission", "public")
            file_hash = metadata.get("file_hash", "unknown")

            if source and source not in qdrant_files:
                qdrant_files[source] = {
                    "permission": perm,
                    "hash": file_hash
                }

    # Qdrant'ta olanları JSON'a ekle (eğer yoksa)
    updated = False
    for filename, info in qdrant_files.items():
        if filename not in registry:
            registry[filename] = {
                "hash": info["hash"],
                "permission": info["permission"],
                "synced_at": str(time.time())
            }
            updated = True

    # JSON'da olanları kontrol et (Qdrant'ta yoksa sil)
    files_to_remove = []
    for filename in registry:
        if filename not in qdrant_files:
            files_to_remove.append(filename)

    for filename in files_to_remove:
        del registry[filename]
        updated = True

    if updated:
        save_registry(registry)

    return registry


# ==============================================================================
# ARAYÜZ - SIDEBAR
# ==============================================================================
with st.sidebar:
    try:
        st.image("bilimp_logo.png", width="stretch")
    except:
        st.warning("Logo Yok")

    # SENKRONIZASYON
    if "registry_synced" not in st.session_state:
        with st.status("🔄 Sistem Başlatılıyor...", expanded=False) as status:
            sync_registry_with_qdrant()
            status.update(label="✅ Sistem Hazır", state="complete", expanded=False)
        st.session_state["registry_synced"] = True

    st.markdown("### 🛠️ Sistem Ayarları")
    if "last_role" not in st.session_state:
        st.session_state.last_role = "admin"

    current_user_role = st.selectbox("👤 Kullanıcı Rolü", ["public", "user", "management", "admin", "private"], index=3)
    if current_user_role != st.session_state.last_role:
        st.session_state.messages = []
        st.session_state.last_role = current_user_role
        st.rerun()

    with st.expander("ℹ️ Yetki Detayı"):
        st.code(get_allowed_permissions(current_user_role))

    st.divider()
    st.markdown("### 🧠 Yapay Zeka Motoru")
    gemini_models_map = {
        "Gemini 2.5 Flash (Hızlı)": "gemini-2.5-flash",
        "Gemini 3.0 Flash (Akıllı + Hızlı)": "gemini-3-flash-preview"
    }
    ollama_list = get_local_ollama_models()
    model_options = list(gemini_models_map.keys())
    if ollama_list:
        model_options.extend([f"Ollama: {m}" for m in ollama_list])
    else:
        model_options.append("Ollama (Model Yok)")

    selected_option = st.selectbox("Model Seçimi", model_options)
    llm_model_id, llm_type = None, "ollama"
    if "Gemini" in selected_option:
        llm_type = "gemini"
        llm_model_id = gemini_models_map[selected_option]
    elif "Ollama" in selected_option:
        llm_type = "ollama"
        llm_model_id = selected_option.split(": ")[1]

    api_key = ""
    if llm_type == "gemini":
        api_key = st.text_input("🔑 Google API Key", type="password")

    st.divider()
    st.markdown("### 🎛️ İnce Ayarlar")
    temperature = st.slider("Yaratıcılık", 0.0, 1.0, 0.3, step=0.1)
    top_k = st.number_input("Bağlam (Chunk)", 1, 20, 5)
    score_threshold = st.slider("Benzerlik Eşiği", 0.0, 0.9, 0.40, step=0.05)
    with st.expander("📄 Chunk Parametreleri"):
        c_size = st.number_input("Boyut", 500, 5000, 2500)
        c_over = st.number_input("Örtüşme", 0, 1000, 200)

st.header("📄 Bilimp Doküman Asistanı (Streaming Agent)")
t1, t2 = st.tabs(["📂 **Belge Yönetimi**", "💬 **Akıllı Sohbet**"])

# --- TAB 1: BELGE YÖNETİMİ ---
with t1:
    col_upload, col_list = st.columns([1, 1], gap="large")
    with col_upload:
        st.markdown("#### ⬆️ Belge Yükle")
        up_file = st.file_uploader("Dosyayı buraya sürükleyin", type=["pdf", "docx", "xlsx", "pptx"],
                                   label_visibility="collapsed")
        if up_file:
            bytes_data = up_file.getvalue()
            f_name = up_file.name
            curr_md5 = calculate_md5(bytes_data)

            # Qdrant'ta bu belge var mı kontrol et
            client = get_qdrant_client()
            file_exists = False
            hash_matches = False

            if client.collection_exists(COLLECTION_NAME):
                scroll_result = client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=1000,
                    with_payload=True
                )

                for point in scroll_result[0]:
                    metadata = point.payload.get("metadata", {})
                    source = metadata.get("source", "")
                    stored_hash = metadata.get("file_hash", "")

                    if source == f_name:
                        file_exists = True
                        if stored_hash == curr_md5:
                            hash_matches = True
                        break

            if file_exists and hash_matches:
                st.warning(f"⚠️ **{f_name}** zaten mevcut (değişiklik yok).")
            elif file_exists and not hash_matches:
                st.info(f"🔄 **{f_name}** önceki versiyonundan farklı. Güncellenecek.")
            else:
                st.success(f"✅ **{f_name}** analize hazır.")

            if st.button("🚀 Sisteme Entegre Et", type="primary"):
                if file_exists and hash_matches:
                    st.info("Belge zaten güncel durumda.")
                else:
                    with st.status("İşleniyor...", expanded=True) as s:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f_name)[1]) as tmp:
                            tmp.write(bytes_data)
                            tmp_path = tmp.name

                        init_collection()
                        delete_by_source(f_name)

                        chunks = []
                        if f_name.endswith(".pptx"):
                            chunks = process_pptx_native(tmp_path, f_name, current_user_role)
                        else:
                            chunks = process_text_based(tmp_path, f_name, c_size, c_over, current_user_role)

                        if chunks:
                            add_documents_to_qdrant(chunks, file_hash=curr_md5)

                            current_reg = load_registry()
                            current_reg[f_name] = {
                                "hash": curr_md5,
                                "permission": current_user_role,
                                "updated_at": str(time.time())
                            }
                            save_registry(current_reg)

                            s.update(label="Tamamlandı!", state="complete", expanded=False)
                            st.toast("Başarılı!", icon="🎉")
                            time.sleep(1)
                            st.rerun()
                        else:
                            s.update(label="Hata", state="error")
                            st.error("Ayrıştırılamadı.")

                        os.unlink(tmp_path)

    with col_list:
        st.markdown("#### 🗂️ Sistemdeki Belgeler")

        client = get_qdrant_client()
        visible_files = []

        if client.collection_exists(COLLECTION_NAME):
            scroll_result = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=True
            )

            unique_files = {}
            for point in scroll_result[0]:
                source = point.payload.get("metadata", {}).get("source", "")
                perm = point.payload.get("metadata", {}).get("permission", "public")
                if source and source not in unique_files:
                    unique_files[source] = perm

            allowed_view_perms = get_allowed_permissions(current_user_role)
            for fname, perm in unique_files.items():
                if perm in allowed_view_perms:
                    visible_files.append((fname, perm))

        if not visible_files:
            st.info("Görüntülenecek belge yok.")
        else:
            for fname, perm in visible_files:
                c1, c2 = st.columns([0.8, 0.2])
                with c1:
                    st.markdown(
                        f"""<div style="padding:10px; background:#161b22; border-radius:8px; margin-bottom:5px; border:1px solid #30363d;"><span style="color:white; font-weight:600;">📄 {fname}</span><span style="background:#238636; color:white; padding:2px 8px; border-radius:4px; font-size:0.8em; margin-left:10px;">{perm}</span></div>""",
                        unsafe_allow_html=True)
                with c2:
                    if st.button("🗑️", key=f"del_{fname}"):
                        delete_document_globally(fname)
                        st.rerun()

# --- TAB 2: SOHBET (STREAMING) ---
with t2:
    def get_formatted_history(messages, max_pairs=5):
        """
        Mesaj geçmişini LangChain formatına çevirir.
        """
        history = []
        all_msgs = messages.copy()
        recent = all_msgs[-(max_pairs * 2):]

        for msg in recent:
            content = msg.get("content", "")
            if not content or content.strip() == "":
                continue

            if msg["role"] == "user":
                history.append(HumanMessage(content=content))
            elif msg["role"] == "assistant":
                history.append(AIMessage(content=content))

        return history


    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and "sources" in m and m["sources"]:
                with st.expander(f"🔍 Referans Kaynaklar ({len(m['sources'])})"):
                    for i, doc in enumerate(m['sources']):
                        score_val = doc.metadata.get("score", 0.0)
                        st.markdown(f"**#{i + 1}** | 📂 `{doc.metadata.get('source')}` | 📊 Skor: `{score_val:.4f}`")
                        st.caption(doc.page_content)
                        st.divider()

    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            client = get_qdrant_client()
            if not client.collection_exists(COLLECTION_NAME):
                st.error("Veritabanı boş.")
            else:
                ready = True
                llm = None
                if llm_type == "gemini":
                    if not api_key:
                        st.error("API Key Eksik!")
                        ready = False
                    else:
                        llm = ChatGoogleGenerativeAI(
                            model=llm_model_id,
                            google_api_key=api_key,
                            temperature=temperature
                        )
                elif llm_type == "ollama":
                    if "Yok" in selected_option:
                        st.error("Model Yok!")
                        ready = False
                    else:
                        llm = ChatOllama(model=llm_model_id, temperature=temperature)

                if ready and llm:
                    try:
                        router_answer_language, _, _, _ = choose_answer_language(prompt, "")
                        router_language_policy = build_language_policy_prompt(router_answer_language)

                        @tool
                        def bilimp_knowledge_base(query: str):
                            """
                            Bilimp AI Asistanı'nın şirket içi bilgi bankasında arama yapmasını sağlar.
                            """
                            pass


                        llm_with_tools = llm.bind_tools([bilimp_knowledge_base])

                        history_messages = st.session_state.messages[:-1]
                        history_langchain_format = get_formatted_history(history_messages, max_pairs=5)

                        identity_section = """
                        Sen profesyonel, yardımsever ve kurumsal bir asistansın.
                        KİMLİĞİN:
                        - Adın: **Bilimp AI Asistanı**.
                        - Görevin: Çalışanlara şirket içi dökümanlar, yönetmelikler ve prosedürler hakkında bilgi sağlamak.
                        
                        YETENEKLERİN VE HAFIZA:
                        - Güçlü bir hafızan var. Sohbet geçmişindeki TÜM mesajları hatırlarsın.
                        - Kullanıcı "Önceki soruma ne cevap verdin?" gibi sorular sorarsa, sohbet geçmişine bakarak cevapla.
                        
                        DAVRANIŞ KURALLARI:
                        1. Eğer kullanıcı "Kimsin?" derse kendini tanıt.
                        2. Başka bir model olduğunu ASLA SÖYLEME.
                        3. Kullanıcıya her zaman nazik ve "siz" diliyle hitap et.
                        4. Hafıza soruları için TOOL KULLANMA, direkt sohbet geçmişinden cevapla.
                        """

                        router_section = """
                        GÖREVİN:
                        Gelen soruyu ve sohbet geçmişini analiz edip 'bilimp_knowledge_base' aracını kullanıp kullanmayacağına karar ver.
                        
                        KARAR MANTIĞI:
                        1. **Veri İsteği:** Şirket verisi, sayı, kural soruluyorsa -> TOOL KULLAN.
                        2. **Takip Sorusu:** "Peki kaç tane?" gibi önceki konunun devamıysa -> TOOL KULLAN.
                        3. **HAFIZA SORULARI:** "Önceki cevabın neydi?" -> TOOL KULLANMA, sohbet geçmişinden cevapla.
                        4. **Sohbet:** "Merhaba" -> TOOL KULLANMA.
                        """

                        router_language_policy = """
                        DİL POLİTİKASI:
                        1. Yanıtınızı kullanıcının sorduğu dille verin.
                        2. Kullanıcı çeviri isterse yalnızca istenen çeviriyi üretin.
                        3. Kullanıcı Türkçe yazdıysa Türkçe, İngilizce yazdıysa İngilizce yanıt verin.
                        4. Bu router aşamasında yalnızca dil ve üslup kurallarını uygula; herhangi bir bağlam yoksa otomatik olarak "bilmiyorum" türü bir yanıt verme.
                        """
                        full_system_prompt = identity_section + "\n\n" + router_section + "\n\n" + router_language_policy

                        input_msgs = [
                                         SystemMessage(content=full_system_prompt)
                                     ] + history_langchain_format + [
                                         HumanMessage(content=prompt)
                                     ]

                        ai_msg = llm_with_tools.invoke(input_msgs)

                        final_response = ""
                        retrieved_docs = []

                        if ai_msg.tool_calls:
                            with st.status("📚 Bilgi Bankası Taranıyor...", expanded=True) as s:
                                dense_emb = get_dense_embeddings()
                                sparse_emb = get_sparse_embeddings()
                                vector_store = QdrantVectorStore(
                                    client=client,
                                    collection_name=COLLECTION_NAME,
                                    embedding=dense_emb,
                                    vector_name="content",
                                    sparse_embedding=sparse_emb,
                                    sparse_vector_name="sparse",
                                    retrieval_mode=RetrievalMode.HYBRID
                                )
                                allowed_perms = get_allowed_permissions(current_user_role)
                                perm_filter = rest_models.Filter(must=[
                                    rest_models.FieldCondition(
                                        key="metadata.permission",
                                        match=rest_models.MatchAny(any=allowed_perms)
                                    )
                                ])

                                results = vector_store.similarity_search_with_score(prompt, k=top_k, filter=perm_filter)
                                for doc, score in results:
                                    if score >= score_threshold:
                                        doc.metadata["score"] = score
                                        retrieved_docs.append(doc)

                                context_str = "\n\n".join([d.page_content for d in retrieved_docs])
                                s.update(label="Bilgiler Getirildi!", state="complete", expanded=False)

                            answer_language, question_language, context_language, language_source = choose_answer_language(
                                prompt,
                                context_str,
                            )
                            language_label = get_language_label(answer_language)

                            rag_system_prompt = f"""
                            SYSTEM INSTRUCTION: You are a helpful assistant.
                            {build_language_policy_prompt(answer_language)}

                            Answer the user's latest question using the FOUND DOCUMENTS below.
                            
                            FOUND DOCUMENTS:
                            {context_str}
                            
                            STRICT RULES:
                            1. Use only the provided documents.
                            2. Consider chat history for continuity.
                            """
                            st.markdown(f"📚 **Dokumanlardan Yanitlaniyor ({language_label})**")
                            st.caption(
                                f"Dil karari: yanit={answer_language}, soru={question_language}, "
                                f"baglam={context_language}, kaynak={language_source}"
                            )

                            rag_messages = [
                                               SystemMessage(content=rag_system_prompt)
                                           ] + history_langchain_format + [
                                               HumanMessage(content=prompt)
                                           ]

                            stream_generator = llm.stream(rag_messages)
                            final_response = st.write_stream(stream_generator)

                        else:
                            raw_content = ai_msg.content
                            content_text = ""

                            if isinstance(raw_content, str):
                                content_text = raw_content
                            elif isinstance(raw_content, list):
                                for item in raw_content:
                                    if isinstance(item, list):
                                        for sub_item in item:
                                            if isinstance(sub_item, dict):
                                                content_text += sub_item.get("text", "")
                                    elif isinstance(item, dict):
                                        content_text += item.get("text", "")
                                    elif isinstance(item, str):
                                        content_text += item
                            else:
                                content_text = str(raw_content)

                            st.markdown("💬 **Sohbet Modu:**")
                            final_response = st.write_stream(stream_text_generator(content_text))

                        if retrieved_docs:
                            with st.expander(f"🔍 Referans Kaynaklar ({len(retrieved_docs)})"):
                                for i, doc in enumerate(retrieved_docs):
                                    score_val = doc.metadata.get("score", 0.0)
                                    st.markdown(
                                        f"**#{i + 1}** | 📂 `{doc.metadata.get('source')}` | 📊 Skor: `{score_val:.4f}`")
                                    st.caption(doc.page_content)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_response,
                            "sources": retrieved_docs
                        })

                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg:
                            st.error("⚠️ API Kotası Doldu.")
                        else:
                            st.error(f"Hata: {e}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Bir hata oluştu: {error_msg}",
                            "sources": []
                        })