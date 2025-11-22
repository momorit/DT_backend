"""FastAPI application exposing a minimal RAG pipeline for Unity clients."""
from __future__ import annotations

import os
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import httpx
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

import chromadb
from chromadb.api.models import Collection
from chromadb.utils import embedding_functions
from duckduckgo_search import DDGS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"
COLLECTION_NAME = "local_text_documents"
CHROMA_DB_DIR = Path(__file__).resolve().parent / "chroma_store"
EMBEDDING_MODEL = "nomic-embed-text" # Kept for reference
CHAT_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1"


# ---------------------------------------------------------------------------
# Embedding Wrapper (new Chroma API compatible)
# ---------------------------------------------------------------------------

class VoyageEmbeddingWrapper(embedding_functions.EmbeddingFunction):
    """EmbeddingFunction wrapper using VoyageAI embedding models."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model_name = "voyage-3"

    def __call__(self, input: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "input": input,
        }

        # Chroma expects a synchronous call. We use httpx.Client for sync requests.
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                "https://api.voyageai.com/v1/embeddings",
                headers=headers,
                json=payload
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"VoyageAI embedding error: {resp.status_code} {resp.text}"
            )

        data = resp.json()
        return [item["embedding"] for item in data["data"]]


# ---------------------------------------------------------------------------
# Build (or rebuild) Chroma collection
# ---------------------------------------------------------------------------

def build_collection(api_key: str) -> Collection:
    """Initialise the Chroma collection and upsert all .txt files."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    embedding_function = VoyageEmbeddingWrapper(api_key=api_key)
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    documents: List[str] = []
    metadatas: List[Dict[str, str]] = []
    ids: List[str] = []

    # Check for documents and prepare for upsert
    if DOCUMENTS_DIR.exists():
        for file_path in sorted(DOCUMENTS_DIR.glob("*.txt")):
            try:
                text = file_path.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                documents.append(text)
                metadatas.append({"filename": file_path.name})
                ids.append(file_path.stem)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if documents:
        # Upsert updates existing IDs and adds new ones
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        print(f"Upserted {len(documents)} documents into collection '{COLLECTION_NAME}'.")

    return collection


# ---------------------------------------------------------------------------
# API Models
# ---------------------------------------------------------------------------

class QueryPayload(BaseModel):
    """Expected structure of POST /query payload from Unity."""
    question: str = Field(..., description="User question that should be answered")
    sensor_data: Dict[str, float] = Field(
        default_factory=dict,
        description="Arbitrary sensor readings forwarded from Unity (e.g. temperature).",
    )


class QueryResponse(BaseModel):
    """Response payload returned to Unity."""
    answer: str
    context: List[str]
    web_results: List[str]
    sensor_data: Dict[str, float]


# ---------------------------------------------------------------------------
# Web Search Helper
# ---------------------------------------------------------------------------

def perform_web_search(query: str, max_results: int = 3) -> List[str]:
    """Perform a web search using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = [r["body"] for r in ddgs.text(query, max_results=max_results)]
        return results
    except Exception as e:
        print(f"Web search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# LLM interaction helpers
# ---------------------------------------------------------------------------

async def call_groq_chat(api_key: str, prompt: str) -> str:
    """Call Groq's chat completion endpoint with the assembled prompt."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "You answer using the provided context."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{GROQ_API_URL}/chat/completions", json=payload, headers=headers
        )
        
    if response.status_code != 200:
        raise RuntimeError(
            f"Groq chat completion failed: {response.status_code} {response.text}"
        )
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def assemble_prompt(question: str, context_chunks: List[str], web_results: List[str], sensor_data: Dict[str, float]) -> str:
    """Combine retrieved document snippets, web results, and sensor readings into one prompt."""
    context_block = "\n\n".join(context_chunks) if context_chunks else "No local context found."
    web_block = "\n\n".join(web_results) if web_results else "No web search results found."
    sensor_lines = "\n".join(f"- {key}: {value}" for key, value in sensor_data.items())
    sensor_block = sensor_lines if sensor_lines else "- (no sensor readings supplied)"
    
    prompt = (
        "You are a professional and helpful assistant. Answer the question using the provided resources.\n"
        "PRIORITY 1: Use the 'Context' (local documents) if it contains the answer.\n"
        "PRIORITY 2: If the answer is not in the Context, use the 'Web Search Results'.\n"
        "PRIORITY 3: If the answer is not in either, use your internal knowledge but state that you are doing so.\n"
        "Do not make up information.\n\n"
        f"Question: {question}\n\n"
        f"Sensor Data:\n{sensor_block}\n\n"
        f"Context (Local Documents):\n{context_block}\n\n"
        f"Web Search Results:\n{web_block}"
    )
    return prompt


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create the FastAPI instance and prepare the RAG components."""
    # Load keys from env
    groq_key = os.environ.get("GROQ_API_KEY")
    voyage_key = os.environ.get("VOYAGE_API_KEY")
    
    if not groq_key:
        print("WARNING: GROQ_API_KEY not set. API calls will fail.")
    if not voyage_key:
        print("WARNING: VOYAGE_API_KEY not set. Embeddings will fail.")

    # Ensure directories exist.
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    # Build vector store from local text files.
    # We pass the key if available, otherwise it might fail inside if used.
    collection = None
    if voyage_key:
        try:
            collection = build_collection(api_key=voyage_key)
        except Exception as e:
            print(f"Failed to build collection: {e}")

    app = FastAPI(title="Mini RAG Backend", version="0.1.0")

    # Allow Unity clients to call API without CORS restrictions.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/query", response_model=QueryResponse)
    async def query_rag(payload: QueryPayload) -> QueryResponse:
        """Handle questions from Unity by retrieving context, searching web, and asking Groq LLM."""
        if not payload.question.strip():
            raise HTTPException(status_code=400, detail="Question must not be empty.")
        
        if not collection:
             raise HTTPException(status_code=503, detail="Vector collection not initialized (check VOYAGE_API_KEY).")
        
        if not groq_key:
             raise HTTPException(status_code=503, detail="LLM service not configured (check GROQ_API_KEY).")

        # 1. Retrieve the most relevant document snippets (Local RAG)
        results = await run_in_threadpool(
            lambda: collection.query(query_texts=[payload.question], n_results=1)
        )
        context_chunks = results.get("documents", [[]])[0]

        # 2. Perform Web Search
        web_results = await run_in_threadpool(
            lambda: perform_web_search(payload.question)
        )

        # 3. Assemble full prompt for LLM.
        prompt = assemble_prompt(
            question=payload.question,
            context_chunks=context_chunks,
            web_results=web_results,
            sensor_data=payload.sensor_data,
        )

        # 4. Ask Groq chat model.
        answer = await call_groq_chat(api_key=groq_key, prompt=prompt)

        # Return everything for Unity debugging.
        return QueryResponse(
            answer=answer,
            context=context_chunks,
            web_results=web_results,
            sensor_data=payload.sensor_data
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

app = create_app()
