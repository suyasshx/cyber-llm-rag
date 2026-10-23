"""
Lightweight production FastAPI application for CyberLLM RAG.
Provides REST endpoints with OpenAPI/Swagger docs for cybersecurity queries and semantic search.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from rag.cloud_rag import CloudCyberRAG

app = FastAPI(
    title="CyberLLM RAG API",
    description="Cybersecurity Threat Intelligence RAG Service powered by MITRE ATT&CK, CWE, CVE, and Sigma",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for external frontends or integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Engine
rag_engine = CloudCyberRAG(
    llm_provider=os.getenv("LLM_PROVIDER", "groq"),
    api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"),
    model_name=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
)


class QueryRequest(BaseModel):
    question: str = Field(..., example="How does Log4Shell (CVE-2021-44228) work and how can it be mitigated?")
    top_k: int = Field(4, ge=1, le=10, description="Number of threat intelligence sources to retrieve")
    source_filter: Optional[List[str]] = Field(None, description="Filter sources: attack, cwe, cve, sigma")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    model: str
    latency_seconds: float


class HealthResponse(BaseModel):
    status: str
    indexed_documents: int
    llm_provider: str
    embedding_model: str


@app.get("/", tags=["Health"])
def root():
    return {
        "message": "CyberLLM RAG API is operational",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(
        status="healthy",
        indexed_documents=len(rag_engine.documents),
        llm_provider=rag_engine.llm_provider,
        embedding_model=rag_engine.embedding_model_name
    )


@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
def query_rag(req: QueryRequest):
    """Query CyberLLM with verified threat intelligence retrieval"""
    try:
        result = rag_engine.query(
            question=req.question,
            top_k=req.top_k,
            source_filter=req.source_filter
        )
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            model=result["model"],
            latency_seconds=result["latency_seconds"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/retrieve", tags=["RAG"])
def retrieve_sources(
    q: str = Query(..., description="Threat search term or query"),
    top_k: int = Query(5, ge=1, le=20)
):
    """Retrieve raw threat intelligence documents matching query"""
    return rag_engine.retrieve(query=q, top_k=top_k)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
