"""
api.py — TÜBİTAK 1505 Hedef Kitle API (bağımsız FastAPI servisi).

Bilimp bu servisi çağırır. Kontrat: API Kontrat Dokümanı v0.1.

Uçlar:
  POST   /api/v1/query                                              → RAG soru-cevap (kullanıcı JWT'si; tenant = user_context.sirket_id)
  POST   /api/v1/documents?sirket_id=..                             → doküman oluştur (içerik + hedef kitle birlikte, servis token'ı)
  PUT    /api/v1/documents/{dokuman_id}/content?sirket_id=..        → doküman içeriğini güncelle (yeni sürüm, servis token'ı)
  DELETE /api/v1/documents/{dokuman_id}?sirket_id=..                → dokümanı sil (servis token'ı)
  PUT    /api/v1/documents/{dokuman_id}/audience?sirket_id=..       → hedef kitle güncelle (servis token'ı)
  POST   /api/v1/documents/audience/bulk?sirket_id=..                → toplu hedef kitle güncelle (servis token'ı, madde 12)
  GET    /api/v1/documents/{dokuman_id}/audience?sirket_id=..       → hedef kitle oku (servis token'ı)
  GET    /api/v1/documents/audience-compliance-report?sirket_id=..&limit=&offset=  → hedef kitle uyum raporu, sayfalı (servis token'ı, madde 16)
  GET    /health                                                     → sağlık kontrolü

Hata Sertleştirme (Faz 4 / madde 15):
  Tüm hata yanıtları {"kod","mesaj","trace_id"} şemasındadır (bkz.
  error_handling.py). Rate limiting bellek-içi bir middleware ile uygulanır
  (bkz. rate_limiting.py, RATE_LIMIT_* ortam değişkenleri). aud uyuşmazlığı
  (yanlış token tipi) 403 döner; diğer kimlik doğrulama hataları 401.

Doküman Yaşam Döngüsü (Faz 2 / madde 2):
  dokuman_id Bilimp tarafından üretilir (dosya adı, tenant içinde benzersiz).
  Sürümleme: içerik değiştiğinde 'versiyon' 1 artar, eski sürümün indeks
  noktaları silinir. İçerik güncellemesi optimistic locking (beklenen_versiyon)
  ile korunur. Ayrıntılı tasarım kararları için document_ingestion_service.py
  başlığına bakınız.

Hedef Kitle Sürümü (Faz 4 / madde 11):
  audience_versiyon, içerik 'versiyon'undan BAĞIMSIZ ayrı bir sayaçtır —
  yalnızca PUT /audience ile artar. PUT /audience, beklenen_audience_versiyon
  mevcut sürümle eşleşmezse 409 döner (gerçek optimistic locking).

Çoklu Hesap (Multi-Tenant) Mimarisi:
  Her şirket (sirket_id) kendi Qdrant koleksiyonuna sahiptir (Seviye C
  fiziksel izolasyon, bkz. tenancy.py). Sorgu ucunda tenant, JWT'nin
  user_context.sirket_id alanından otomatik belirlenir. Servis token'ı
  bireysel bir kullanıcıya bağlı olmadığından, yönetim uçlarında tenant
  açık bir sirket_id sorgu parametresiyle belirtilmelidir.

Yapılandırma (ortam değişkenleri — .env üzerinden):
  QDRANT_URL, QDRANT_COLLECTION
  JWT_ALGORITHM, JWT_ISSUER, JWT_AUDIENCE_QUERY, JWT_AUDIENCE_ADMIN,
  JWT_CLOCK_SKEW_SECONDS, JWT_PUBLIC_KEYS_JSON / JWT_PUBLIC_KEY
  (bkz. auth.py başlığı — ayrıntılı açıklama orada)
  LLM_PROVIDER = "gemini" | "ollama"
  GOOGLE_API_KEY (gemini için) | OLLAMA_MODEL (ollama için)

ESKİ NOT: Bu dosyanın önceki hâlindeki Department/Rank/User + hardcoded
'postgresql://postgres:12345@localhost:5432/testDB' bağlantısı KALDIRILDI.
Sırlar artık ortam değişkenlerinden okunur; hiçbir kimlik bilgisi koda gömülü
değildir.
"""

from __future__ import annotations
import os
import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status, Path, Body, Query

from qdrant_client import QdrantClient
from langchain_core.messages import HumanMessage, AIMessage

from abac import UserContext
from auth import verify_user_context, verify_service_token, ServiceIdentity
from tenancy import ConventionTenantRegistry, resolve_tenant_collection
from api_schemas import (
    QueryRequest, QueryResponse, QuerySource,
    AudienceUpdateRequest, AudienceUpdateResponse,
    AudienceGetResponse, AudienceComplianceReportResponse, ComplianceReportItem,
    DocumentCreateRequest, DocumentCreateResponse,
    DocumentContentUpdateRequest, DocumentContentUpdateResponse,
    DocumentDeleteResponse,
    BulkAudienceUpdateRequest, BulkAudienceUpdateResponse, BulkAudienceUpdateItemResult,
)
from document_ingestion_service import (
    create_document, update_document_content, delete_document,
    DocumentIngestionError,
)
from audience_service import (
    update_document_audience, get_document_audience,
    find_documents_without_audience, AudienceUpdateError,
    bulk_update_document_audience, BulkAudienceUpdateItem,
)
from rag_service import run_rag_query
from error_handling import install_error_handling
from rate_limiting import install_rate_limiting


# ── Yapılandırma ──────────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "Tubitak_Dokumanlar_Hybrid")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
THRESHOLD = float(os.getenv("RAG_THRESHOLD", "0.3"))

# Seviye C: şirket başına ayrı Qdrant koleksiyonu (bkz. tenancy.py). DIP
# gereği çağıranlar somut Qdrant detaylarına değil bu soyutlamaya bağımlıdır;
# ileride farklı bir TenantRegistry implementasyonuna geçmek (örn. veritabanı
# destekli) yalnızca bu satırın değişmesini gerektirir.
_tenant_registry = ConventionTenantRegistry()


# ── Ağır kaynaklar: uygulama ömrü boyunca tek sefer kurulur ───────────────────
_resources: dict = {}


def _build_llm():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.0,
        )
    from langchain_ollama import ChatOllama
    # ÖNEMLİ: Bu kod bir Docker container'ı içinde çalışır. "localhost" burada
    # container'ın kendisini işaret eder, host makineyi DEĞİL — bu yüzden host'ta
    # çalışan Ollama'ya varsayılan base_url ile ulaşılamaz. Docker Desktop'ın
    # host'a özel adresi kullanılır.
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "gemma3:12b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
        temperature=0.0,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import FastEmbedSparse
    import torch

    _resources["client"] = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _resources["dense"] = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBED_MODEL", "ytu-ce-cosmos/turkish-e5-large"),
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    _resources["sparse"] = FastEmbedSparse(model_name="Qdrant/bm25")
    _resources["llm"] = _build_llm()
    yield
    _resources.clear()


app = FastAPI(title="TÜBİTAK 1505 Hedef Kitle API", version="0.3", lifespan=lifespan)
install_error_handling(app)   # (Faz 4 / madde 15) standart hata gövdesi + trace_id
install_rate_limiting(app)    # (Faz 4 / madde 15) istek sınırlama


def _client() -> QdrantClient:
    client = _resources.get("client")
    if client is None:
        raise HTTPException(500, "Servis henüz hazır değil.")
    return client


# ══════════════════════════════════════════════════════════════════════════════
# UÇ NOKTA 1 — Erişim Sorgu (RAG)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/v1/query", response_model=QueryResponse, tags=["Sorgu"])
def query_endpoint(
    req: QueryRequest = Body(...),
    user: UserContext = Depends(verify_user_context),   # imzalı JWT'den gelir
):
    if not req.soru or not req.soru.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Soru boş olamaz.")

    # Çok turlu konuşma (Faz 3): önceki turlar LangChain mesajlarına çevrilir.
    # gecmis_mesajlar boşsa history=[] → run_rag_query tek turluk davranışa
    # (Faz 3 öncesiyle birebir aynı) düşer.
    history = [
        HumanMessage(content=m.icerik) if m.rol == "kullanici" else AIMessage(content=m.icerik)
        for m in req.gecmis_mesajlar
    ]

    result = run_rag_query(
        llm=_resources["llm"],
        client=_client(),
        tenant_registry=_tenant_registry,
        dense_embeddings=_resources["dense"],
        sparse_embeddings=_resources["sparse"],
        user=user,
        question=req.soru,
        top_k=TOP_K,
        threshold=THRESHOLD,
        history=history,
    )

    if result.durum == "hata":
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, result.yanit)

    kaynaklar = [
        QuerySource(
            dokuman_id=d.metadata.get("source", ""),
            dosya_adi=d.metadata.get("source", ""),
            sayfa=d.metadata.get("page"),
            versiyon=d.metadata.get("versiyon"),
            skor=float(d.metadata.get("score", 0.0)),
        )
        for d in result.kaynaklar
    ]
    return QueryResponse(durum=result.durum, yanit=result.yanit, kaynaklar=kaynaklar)


# ══════════════════════════════════════════════════════════════════════════════
# UÇ NOKTA — Doküman Oluştur (içerik + hedef kitle politikası TEK istekte)
# ══════════════════════════════════════════════════════════════════════════════
@app.post(
    "/api/v1/documents",
    response_model=DocumentCreateResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Doküman Yaşam Döngüsü"],
)
def create_document_endpoint(
    sirket_id: int = Query(..., description="Dokümanın yükleneceği tenant (şirket) kimliği"),
    req: DocumentCreateRequest = Body(...),
    _service: ServiceIdentity = Depends(verify_service_token),   # yalnızca Bilimp servis token'ı
):
    collection = resolve_tenant_collection(_tenant_registry, sirket_id)
    file_ext = os.path.splitext(req.dokuman_id)[1].lower()

    try:
        result = create_document(
            client=_client(),
            collection=collection,
            dokuman_id=req.dokuman_id,
            file_bytes=base64.b64decode(req.dosya_icerigi_base64),
            file_ext=file_ext,
            audience_policy=req.audience_policy,
            dense_embeddings=_resources["dense"],
            sparse_embeddings=_resources["sparse"],
        )
    except DocumentIngestionError as e:
        if e.code == "already_exists":
            raise HTTPException(status.HTTP_409_CONFLICT, e.message)
        if e.code == "too_large":
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, e.message)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, e.message)

    return DocumentCreateResponse(
        durum="basarili", dokuman_id=result.dokuman_id,
        versiyon=result.versiyon, chunk_sayisi=result.chunk_sayisi,
    )


# ══════════════════════════════════════════════════════════════════════════════
# UÇ NOKTA — Doküman İçeriğini Güncelle (Sürümleme)
# ══════════════════════════════════════════════════════════════════════════════
@app.put(
    "/api/v1/documents/{dokuman_id}/content",
    response_model=DocumentContentUpdateResponse,
    tags=["Doküman Yaşam Döngüsü"],
)
def update_document_content_endpoint(
    dokuman_id: str = Path(...),
    sirket_id: int = Query(..., description="Dokümanın ait olduğu tenant (şirket) kimliği"),
    req: DocumentContentUpdateRequest = Body(...),
    _service: ServiceIdentity = Depends(verify_service_token),
):
    collection = resolve_tenant_collection(_tenant_registry, sirket_id)
    file_ext = os.path.splitext(dokuman_id)[1].lower()

    try:
        result = update_document_content(
            client=_client(),
            collection=collection,
            dokuman_id=dokuman_id,
            file_bytes=base64.b64decode(req.dosya_icerigi_base64),
            file_ext=file_ext,
            beklenen_versiyon=req.beklenen_versiyon,
            dense_embeddings=_resources["dense"],
            sparse_embeddings=_resources["sparse"],
        )
    except DocumentIngestionError as e:
        if e.code == "not_found":
            raise HTTPException(status.HTTP_404_NOT_FOUND, e.message)
        if e.code == "version_conflict":
            raise HTTPException(status.HTTP_409_CONFLICT, e.message)
        if e.code == "too_large":
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, e.message)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, e.message)

    return DocumentContentUpdateResponse(
        durum="basarili", dokuman_id=result.dokuman_id,
        yeni_versiyon=result.versiyon, chunk_sayisi=result.chunk_sayisi,
    )


# ══════════════════════════════════════════════════════════════════════════════
# UÇ NOKTA — Doküman Sil
# ══════════════════════════════════════════════════════════════════════════════
@app.delete(
    "/api/v1/documents/{dokuman_id}",
    response_model=DocumentDeleteResponse,
    tags=["Doküman Yaşam Döngüsü"],
)
def delete_document_endpoint(
    dokuman_id: str = Path(...),
    sirket_id: int = Query(..., description="Dokümanın ait olduğu tenant (şirket) kimliği"),
    _service: ServiceIdentity = Depends(verify_service_token),
):
    collection = resolve_tenant_collection(_tenant_registry, sirket_id)
    try:
        silinen = delete_document(_client(), collection, dokuman_id)
    except DocumentIngestionError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, e.message)

    return DocumentDeleteResponse(
        durum="basarili", dokuman_id=dokuman_id, silinen_nokta_sayisi=silinen,
    )


# ══════════════════════════════════════════════════════════════════════════════
# UÇ NOKTA 2 — Hedef Kitle Güncelle
# ══════════════════════════════════════════════════════════════════════════════
@app.put(
    "/api/v1/documents/{dokuman_id}/audience",
    response_model=AudienceUpdateResponse,
    tags=["Hedef Kitle"],
)
def update_audience_endpoint(
    dokuman_id: str = Path(..., description="Dosya adı (metadata.source)"),
    sirket_id: int = Query(..., description="Dokümanın ait olduğu tenant (şirket) kimliği"),
    req: AudienceUpdateRequest = Body(...),
    _service: ServiceIdentity = Depends(verify_service_token),   # yalnızca Bilimp servis token'ı
):
    collection = resolve_tenant_collection(_tenant_registry, sirket_id)
    try:
        yeni_versiyon = update_document_audience(
            client=_client(),
            collection=collection,
            source=dokuman_id,
            policy=req.audience_policy,
            degistiren_kullanici_id=req.degistiren_kullanici_id,
            beklenen_audience_versiyon=req.beklenen_audience_versiyon,
            allow_empty=req.allow_empty,
        )
    except AudienceUpdateError as e:
        if e.code == "not_found":
            raise HTTPException(status.HTTP_404_NOT_FOUND, e.message)
        if e.code == "empty_rules":
            raise HTTPException(status.HTTP_400_BAD_REQUEST, e.message)
        if e.code == "version_conflict":
            raise HTTPException(status.HTTP_409_CONFLICT, e.message)
        raise HTTPException(status.HTTP_409_CONFLICT, e.message)

    return AudienceUpdateResponse(
        durum="basarili",
        dokuman_id=dokuman_id,
        guncellenen_kural_sayisi=len(req.audience_policy.rules),
        degistiren_kullanici_id=req.degistiren_kullanici_id,
        yeni_audience_versiyon=yeni_versiyon,
        qdrant_sync="tamamlandi",
    )


# ══════════════════════════════════════════════════════════════════════════════
# EK UÇ — Toplu Hedef Kitle Güncelleme (madde 12)
# ══════════════════════════════════════════════════════════════════════════════
@app.post(
    "/api/v1/documents/audience/bulk",
    response_model=BulkAudienceUpdateResponse,
    tags=["Hedef Kitle"],
)
def bulk_update_audience_endpoint(
    sirket_id: int = Query(..., description="Dokümanların ait olduğu tenant (şirket) kimliği"),
    req: BulkAudienceUpdateRequest = Body(...),
    _service: ServiceIdentity = Depends(verify_service_token),   # yalnızca Bilimp servis token'ı
):
    collection = resolve_tenant_collection(_tenant_registry, sirket_id)

    items = [
        BulkAudienceUpdateItem(
            dokuman_id=g.dokuman_id,
            policy=g.audience_policy,
            beklenen_audience_versiyon=g.beklenen_audience_versiyon,
            degistiren_kullanici_id=req.degistiren_kullanici_id,
            allow_empty=g.allow_empty,
        )
        for g in req.guncellemeler
    ]
    sonuclar = bulk_update_document_audience(_client(), collection, items)

    basarili = sum(1 for r in sonuclar if r.basarili)
    return BulkAudienceUpdateResponse(
        durum="tamamlandi",
        toplam=len(sonuclar),
        basarili=basarili,
        basarisiz=len(sonuclar) - basarili,
        sonuclar=[
            BulkAudienceUpdateItemResult(
                dokuman_id=r.dokuman_id,
                durum="basarili" if r.basarili else "hata",
                yeni_audience_versiyon=r.yeni_audience_versiyon,
                hata_kodu=r.hata_kodu,
                mesaj=r.mesaj,
            )
            for r in sonuclar
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
# EK UÇ — Hedef Kitle Oku (kontrat Bölüm 9)
# ══════════════════════════════════════════════════════════════════════════════
@app.get(
    "/api/v1/documents/{dokuman_id}/audience",
    response_model=AudienceGetResponse,
    tags=["Hedef Kitle"],
)
def get_audience_endpoint(
    dokuman_id: str = Path(...),
    sirket_id: int = Query(..., description="Dokümanın ait olduğu tenant (şirket) kimliği"),
    _service: ServiceIdentity = Depends(verify_service_token),   # yalnızca Bilimp servis token'ı
):
    from abac import AudiencePolicy

    collection = resolve_tenant_collection(_tenant_registry, sirket_id)
    info = get_document_audience(_client(), collection, dokuman_id)
    if info is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"'{dokuman_id}' bulunamadı.")
    return AudienceGetResponse(
        dokuman_id=dokuman_id,
        audience_policy=AudiencePolicy(**info.policy) if info.policy else AudiencePolicy(),
        audience_versiyon=info.audience_versiyon,
    )


# ══════════════════════════════════════════════════════════════════════════════
# EK UÇ — Hedef Kitle Uyum Raporu (madde 11 / Kontrat Dokümanı Uç Nokta 4)
# ══════════════════════════════════════════════════════════════════════════════
@app.get(
    "/api/v1/documents/audience-compliance-report",
    response_model=AudienceComplianceReportResponse,
    tags=["Uyum İzleme"],
)
def audience_compliance_report_endpoint(
    sirket_id: int = Query(..., description="Raporun istendiği tenant (şirket) kimliği"),
    limit: int = Query(50, ge=1, le=200, description="Sayfa başına eleman sayısı (Faz 4 / madde 16)"),
    offset: int = Query(0, ge=0, description="Atlanacak eleman sayısı"),
    _service: ServiceIdentity = Depends(verify_service_token),   # yalnızca Bilimp servis token'ı
):
    collection = resolve_tenant_collection(_tenant_registry, sirket_id)
    sayfa, toplam = find_documents_without_audience(_client(), collection, limit=limit, offset=offset)
    return AudienceComplianceReportResponse(
        politikasiz_dokuman_sayisi=toplam,
        limit=limit,
        offset=offset,
        dokumanlar=[
            ComplianceReportItem(dokuman_id=d.dokuman_id, versiyon=d.versiyon)
            for d in sayfa
        ],
    )


@app.get("/health", tags=["Sistem"])
def health():
    try:
        _client().get_collections()
        return {"durum": "saglikli"}
    except Exception:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Qdrant erişilemiyor.")