import os
import time
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Mevcut RAG modüllerini içe aktar
# TEST.py içindeki fonksiyonları kullanacağız.
# Ancak TEST.py bir script olduğu için import ederken çalışmasını istemeyiz.
# Bu yüzden TEST.py'yi biraz refactor etmek gerekebilir veya doğrudan buraya taşıyabiliriz.
# Şimdilik TEST.py'deki mantığı buraya kopyalayarak entegre ediyoruz.

# --- RAG IMPORTS ---
import torch
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from language_utils import choose_answer_language, build_language_policy_prompt, get_no_answer_message

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
    answer_language: str
    question_language: str
    context_language: str
    language_source: str

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
    
    # Rol bazlı erişim kontrolü filtresi
    # Not: Gerçek senaryoda bu mapping daha detaylı olabilir.
    # Şimdilik basitçe: gelen rol, metadatadaki permission ile eşleşmeli.
    # Veya hiyerarşik bir yapı kurulabilir (Admin her şeyi görür vs).
    # Basitlik için birebir eşleşme veya 'admin' ise her şeyi görme mantığı eklenebilir.
    
    # Eğer Admin değilse yetki filtresi uygula (Örnek mantık)
    # Ancak mevcut veri yapısında permission alanı var. 
    # TEST.py'deki mantığı koruyarak direk filtreliyoruz.
    must_conditions.append(
        models.FieldCondition(
            key="metadata.permission",
            match=models.MatchValue(value=request.role)
        )
    )

    if request.doc_type:
        must_conditions.append(
            models.FieldCondition(
                key="metadata.file_type",
                match=models.MatchValue(value=request.doc_type)
            )
        )

    if must_conditions:
        search_kwargs["filter"] = models.Filter(must=must_conditions)

    # 2. Vektör Araması
    try:
        docs_with_scores = vector_store.similarity_search_with_score(
            request.question,
            **search_kwargs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

    # 3. Sonuçları Filtrele (Score Threshold)
    SCORE_THRESHOLD = 0.45
    filtered_docs = [doc for doc, score in docs_with_scores if score >= SCORE_THRESHOLD]
    
    context_parts = [doc.page_content for doc in filtered_docs]
    context_text = "\n\n---\n\n".join(context_parts)
    answer_language, question_language, context_language, language_source = choose_answer_language(
        request.question,
        context_text,
    )

    if not context_text:
        return QueryResponse(
            answer=get_no_answer_message(answer_language),
            context_used=[],
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
            answer_language=answer_language,
            question_language=question_language,
            context_language=context_language,
            language_source=language_source,
        )

    # 4. LLM Üretimi
    prompt_template = """You are a helpful AI assistant.
    {language_policy}

    Use only the context below to answer the user's question.
    If the answer is not present in the context, do not hallucinate.

    Context (Retrieved from Vector Database):
    {context}

    User Question:
    {question}

    Answer:"""
    
    chain = ChatPromptTemplate.from_template(prompt_template) | llm
    
    response_text = chain.invoke({
        "language_policy": build_language_policy_prompt(answer_language),
        "context": context_text,
        "question": request.question
    })

    # Ollama bazen string, bazen obje dönebilir, kontrol edelim.
    final_answer = response_text if isinstance(response_text, str) else str(response_text)

    elapsed_time = (time.perf_counter() - start_time) * 1000
    
    return QueryResponse(
        answer=final_answer,
        context_used=context_parts,
        processing_time_ms=elapsed_time,
        answer_language=answer_language,
        question_language=question_language,
        context_language=context_language,
        language_source=language_source,
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
