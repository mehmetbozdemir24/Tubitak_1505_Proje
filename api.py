import os
import time
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import torch
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document as LangChainDocument
from hallucination_validator import HallucinationValidator

# --- APP SETUP ---
app = FastAPI(title="Tubitak 1505 RAG API", version="1.0.0")

# CORS Setup (Frontend erişimi için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Geliştirme aşamasında hepsi açık
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "Tubitak_Dokumanlar"
EMBEDDING_MODEL_NAME = "ytu-ce-cosmos/turkish-e5-large"
OLLAMA_MODEL_NAME = "gemma3:12b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 API Başlatılıyor... Mod: {DEVICE}")

# --- GLOBAL OBJECTS ---
# Bunları startup event'inde veya global olarak yükleyebiliriz.
client = QdrantClient(url=QDRANT_URL)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

llm = OllamaLLM(
    model=OLLAMA_MODEL_NAME,
    temperature=0.1,
    top_p=0.9,
    repeat_penalty=1.1,
    num_predict=1024,
)

# --- MODELS ---
class QueryRequest(BaseModel):
    question: str
    role: str  # admin, manager, user
    doc_type: Optional[str] = None
    k: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: str
    context_used: List[str]
    processing_time_ms: float

# --- ROUTES ---

@app.get("/health")
def health_check():
    return {"status": "active", "device": DEVICE, "model": OLLAMA_MODEL_NAME}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    start_time = time.perf_counter()
    
    # 1. Filtreleri Hazırla
    search_kwargs = {"k": request.k}
    must_conditions = []
    
    must_conditions.append(
        qdrant_models.FieldCondition(
            key="metadata.permission",
            match=qdrant_models.MatchValue(value=request.role)
        )
    )

    if request.doc_type:
        must_conditions.append(
            qdrant_models.FieldCondition(
                key="metadata.file_type",
                match=qdrant_models.MatchValue(value=request.doc_type)
            )
        )

    if must_conditions:
        search_kwargs["filter"] = qdrant_models.Filter(must=must_conditions)

    # 2. Vektör Araması
    try:
        docs_with_scores = vector_store.similarity_search_with_score(
            request.question,
            **search_kwargs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

    SCORE_THRESHOLD = 0.70
    filtered_docs = [(doc, score) for doc, score in docs_with_scores if score >= SCORE_THRESHOLD]
    
    if not filtered_docs:
        return QueryResponse(
            answer="Bu konuda belgelerimde yeterli kalitede bilgi bulunamadı. Lütfen sorunuzu farklı şekilde ifade edin.",
            context_used=[],
            processing_time_ms=(time.perf_counter() - start_time) * 1000
        )
    
    if len(filtered_docs) < 2:
        return QueryResponse(
            answer="Bu konuda çok az bilgi var. Lütfen daha spesifik bir soru sorun.",
            context_used=[doc.page_content for doc, _ in filtered_docs],
            processing_time_ms=(time.perf_counter() - start_time) * 1000
        )
    
    context_parts = [doc.page_content for doc, _ in filtered_docs]
    context_text = "\n\n---\n\n".join(context_parts)

    # 4. LLM Üretimi
    prompt_template = """Sen yardımcı bir yapay zeka asistanısın. Aşağıdaki bağlam bilgisini kullanarak kullanıcının sorusunu cevapla.
    Eğer bağlamda cevabı bulamazsan, uydurma, sadece "Bilgim yok" de.

    Bağlam (Veritabanından Gelen Bilgi):
    {context}

    Kullanıcı Sorusu:
    {question}

    KATÎ KURALLAR:
    1. SADECE yukarıdaki bağlamdan cevap ver.
    2. Bağlamda cevap YOKSA: "Bu konuda belgelerimde bilgi bulunmuyor" de.
    3. ASLA tahmin yapma, spekülasyon etme veya kendi bilgini kullanma.
    4. Belirsizlik varsa açıkça belirt.
    5. Her cevabın sonunda kaynak belirt.

    Cevap:"""
    
    chain = ChatPromptTemplate.from_template(prompt_template) | llm
    
    response_text = chain.invoke({
        "context": context_text,
        "question": request.question
    })

    final_answer = response_text if isinstance(response_text, str) else str(response_text)
    
    source_docs = [LangChainDocument(page_content=cp) for cp in context_parts]
    
    is_valid, validation_msg = HallucinationValidator.validate_response(
        request.question, final_answer, source_docs
    )
    
    if not is_valid:
        return QueryResponse(
            answer=f"Yanıt kalite kontrolünden geçemedi: {validation_msg}. Lütfen sorunuzu farklı şekilde ifade edin.",
            context_used=context_parts,
            processing_time_ms=(time.perf_counter() - start_time) * 1000
        )

    elapsed_time = (time.perf_counter() - start_time) * 1000
    
    return QueryResponse(
        answer=final_answer,
        context_used=context_parts,
        processing_time_ms=elapsed_time
    )

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    role: str = Form(...)  # Upload eden kişinin rolü (yetki kontrolü için)
):
    start_time = time.perf_counter()
    
    # 1. Dosyayı Kaydet
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save error: {str(e)}")

    # 2. Dosyayı İşle (Chunking)
    from chunker_module import process_file_wrapper
    
    # KULLANICI İSTEĞİ: Yükleyen kişinin rolü neyse, dosya yetkisi o olsun.
    chunks, error = process_file_wrapper(file_path, fixed_permission=role)
    
    if error:
        os.remove(file_path) # Hatalı dosyayı sil
        raise HTTPException(status_code=400, detail=f"Processing error: {error}")
        
    if not chunks:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="No content extracted from file.")

    # 3. Embedding & Vector Store (Qdrant)
    try:
        # UUID ata
        from uuid import uuid4
        chunk_ids = [str(uuid4()) for _ in chunks]
        
        # Qdrant'a ekle
        vector_store.add_documents(documents=chunks, ids=chunk_ids)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector store error: {str(e)}")

    elapsed_time = (time.perf_counter() - start_time) * 1000

    return {
        "filename": file.filename,
        "chunks_count": len(chunks),
        "status": "success",
        "message": f"File processed and indexed in {elapsed_time:.2f} ms"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
