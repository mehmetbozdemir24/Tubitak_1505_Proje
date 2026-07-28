"""
vllm_ingestion.py — Docling + vLLM (VLM görsel açıklama + contextual bağlam)
tabanlı zengin belge → chunk pipeline'ı.

KÖKEN: Bu modül, bilimp_vllm.py'nin (Streamlit demo) INGESTION PIPELINE
bölümünün (eski chunker.py'nin yerini alan, "TEK MODEL mimarisi" — Docling
dönüşümü + VLM görsel açıklama + contextual bağlam üretiminin hepsi aynı
vLLM modelinden geçtiği pipeline) API tarafına (document_ingestion_service.py)
taşınmış hâlidir. Mantık BİREBİR aynıdır; tek fark: vLLM adresi/model adı
artık bilimp_vllm.py'nin WSL'e özgü _detect_host_ip() sihirbazlığı yerine,
api.py::_build_llm() ile TUTARLI ortam değişkenlerinden (VLLM_BASE_URL,
VLLM_MODEL, VLLM_API_KEY) okunur — servis Docker içinde de çalışabilmelidir.

Chunk kuralları (değişmedi):
  * bölüm chunk_size'a sığıyorsa BÜTÜN kalır (başlıktan başlığa)
  * tablolar ASLA bölünmez; metin kesimi Madde sınırlarında
  * breadcrumb her parçaya kopyalanır; kırpıklar komşusuna yapıştırılır
"""
from __future__ import annotations
import os
import re
import io
import json
import time
import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling_core.types.doc import PictureItem


# ══════════════════════════════════════════════════════════════════════════════
# vLLM yapılandırması — api.py::_build_llm() ile AYNI ortam değişkenleri
# ══════════════════════════════════════════════════════════════════════════════
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://host.docker.internal:8000/v1")
VLLM_MODEL_ID = os.getenv("VLLM_MODEL", "google/gemma-4-12B-it")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
CTX_MODEL = VLLM_MODEL_ID  # cache-key'de kullanılır; model değişince eski cache otomatik geçersizleşir

# (v2.1 düzeltmesi) bilimp_vllm.py'de bu AcceleratorDevice.CUDA olarak SABİT
# kodlanmıştı — CUDA'sız hiçbir ortamda (bu sunucu dahil, GPU yoksa) Docling
# AcceleratorDeviceNotAvailableError ile çöküyordu. AUTO: GPU varsa onu
# kullanır (üretimdeki performansı KORUR), yoksa CPU'ya sessizce düşer.
_ACCELERATOR_DEVICE_MAP = {
    "auto": AcceleratorDevice.AUTO, "cuda": AcceleratorDevice.CUDA,
    "cpu": AcceleratorDevice.CPU, "mps": AcceleratorDevice.MPS, "xpu": AcceleratorDevice.XPU,
}
DOCLING_DEVICE = _ACCELERATOR_DEVICE_MAP.get(
    os.getenv("DOCLING_DEVICE", "auto").lower(), AcceleratorDevice.AUTO
)

OUTPUT_DIR = os.getenv("DOCLING_OUTPUT_DIR", "output-docling")  # md + figürler + kontrol dosyaları
CTX_CACHE_FILE = os.getenv("CTX_CACHE_FILE", "contextual_cache.json")
CTX_DOC_LIMIT = 12000
CTX_PARALLEL = int(os.getenv("CTX_PARALLEL", "8"))  # vLLM'e eşzamanlı bağlam isteği sayısı
IMAGE_SCALE = 2.0
USE_VLM_FOR_PICTURES = True
MIN_PICTURE_PX = 80
TR_SPACING_ESIK = 15
MIN_PIECE_LEN = 300  # split-sonrası kırpık eşiği

_HEADERS = [("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")]
_CRUMB_RE = re.compile(r"^\*\*BAĞLAM:\*\*.*$", re.MULTILINE)
_CAPTION_RE = re.compile(r"^(Şekil|Çizelge|Tablo|Figure|Table)\s*[-.]?\s*\d", re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
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
    ad = orijinal_ad or os.path.basename(input_file)
    stem = os.path.splitext(ad)[0]
    os.makedirs(output_dir, exist_ok=True)
    crop_dir = os.path.join(output_dir, stem + "_figures")
    output_md = os.path.join(output_dir, stem + ".md")
    os.makedirs(crop_dir, exist_ok=True)

    def _convert(ocr: bool):
        o = PdfPipelineOptions()
        o.do_table_structure = True
        o.do_formula_enrichment = True
        o.generate_picture_images = True
        o.images_scale = IMAGE_SCALE
        o.do_ocr = ocr
        o.accelerator_options = AcceleratorOptions(device=DOCLING_DEVICE)
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
            if _fitz_uz < 200:  # gerçekten metinsiz -> OCR aç
                doc = _convert(ocr=True)
        except Exception:
            pass

    desc_list = []
    if USE_VLM_FOR_PICTURES:
        # 1) Önce TÜM görselleri topla ve diske kaydet (hızlı, yerel iş)
        gorseller = []  # (pic_no, png_b64) — sıra korunur
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
        import pymupdf4llm
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


# ── contextual retrieval (vLLM, cache'li) ──
def _ctx_cache_load():
    if os.path.exists(CTX_CACHE_FILE):
        with open(CTX_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _ctx_cache_save(c):
    with open(CTX_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False)


def ensure_ctx_model():
    """Ingestion vLLM (gemma-4) kullanıyor → vLLM ayakta mı ve doğru model
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
    çok yavaştı)."""
    cache = _ctx_cache_load()
    n = len(chunks)
    keys = [hashlib.md5(
        (CTX_MODEL + "||" + source + "||" + d.page_content).encode("utf-8")
    ).hexdigest() for d in chunks]

    ctxs = [cache.get(k) or "" for k in keys]  # cache'ten gelenler
    miss = [i for i, c in enumerate(ctxs) if not c]  # üretilmesi gerekenler
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
        _ctx_cache_save(cache)  # tek seferde diske yaz

    added = 0
    for doc, ctx in zip(chunks, ctxs):
        if ctx:
            doc.metadata["context"] = ctx
            doc.page_content = f"{ctx}\n\n{doc.page_content}"
            added += 1
    return chunks, added


# ══════════════════════════════════════════════════════════════════════════════
# Tek giriş noktası — document_ingestion_service.py bunu çağırır
# ══════════════════════════════════════════════════════════════════════════════
def ingest_file(path: str, source: str, chunk_size: int, chunk_overlap: int,
                 audience: dict | None = None) -> list[Document]:
    """
    Diskteki bir dosyayı (pdf/docx/pptx/xlsx) uçtan uca işler: Docling
    dönüşümü + VLM görsel açıklama → tablo-atomik/başlık-tabanlı chunklama →
    contextual bağlam üretimi (vLLM). document_ingestion_service.py'nin
    eski chunker.py çağrısının YERİNİ alır (tüm formatlar TEK pipeline'dan
    geçer — chunk_pptx/chunk_text ayrımı kalkar, Docling hepsini kapsar).

    ensure_ctx_model() burada BİLEREK çağrılmaz: bu, çağıran katmanın
    (document_ingestion_service.py) sorumluluğu — vLLM ayakta değilse
    kullanıcıya HIZLI (dosya işlenmeden ÖNCE) bir hata dönmek için.
    """
    full_md = belge_to_md(path, OUTPUT_DIR, orijinal_ad=source)
    chunks = chunk_md(full_md, source, chunk_size, chunk_overlap, audience)
    if not chunks:
        return []
    chunks, _ctx_added = add_contextual(chunks, full_md, source)
    return chunks
