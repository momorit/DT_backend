"""FastAPI application exposing a minimal RAG pipeline for Unity clients."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import chromadb
from chromadb.api.models import Collection


# Directories and model names used for document ingestion and Groq calls.
DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"
COLLECTION_NAME = "local_text_documents"
CHROMA_DB_DIR = Path(__file__).resolve().parent / "chroma_store"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "llama3-70b-8192"
GROQ_API_URL = "https://api.groq.com/openai/v1"


class GroqEmbeddingFunction:
    """Embedding helper that calls Groq's embedding endpoint via HTTP."""

    def __init__(self, api_key: str, model: str) -> None:
        # Store API credentials and reuse a requests session for efficiency.
        self._api_key = api_key
        self._model = model
        self._session = requests.Session()
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Groq."""
        payload = {"model": self._model, "input": texts}
        response = self._session.post(
            f"{GROQ_API_URL}/embeddings", json=payload, headers=self._headers, timeout=30
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Groq embedding request failed: {response.status_code} {response.text}"
            )
        data = response.json()
        return [item["embedding"] for item in data["data"]]


def build_collection(api_key: str) -> Collection:
    """Initialise the Chroma collection and (re)ingest all .txt files."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # We recreate the collection on every boot to keep the index in sync with the txt files.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    embedding_function = GroqEmbeddingFunction(api_key=api_key, model=EMBEDDING_MODEL)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    documents: List[str] = []
    metadatas: List[Dict[str, str]] = []
    ids: List[str] = []

    for file_path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        # Read each local text file, strip whitespace and store metadata.
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(text)
        metadatas.append({"filename": file_path.name})
        ids.append(file_path.stem)

    if documents:
        # Chroma automatically requests embeddings through our GroqEmbeddingFunction.
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    return collection


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
    sensor_data: Dict[str, float]


def call_groq_chat(api_key: str, prompt: str) -> str:
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
    response = requests.post(
        f"{GROQ_API_URL}/chat/completions", json=payload, headers=headers, timeout=60
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Groq chat completion failed: {response.status_code} {response.text}"
        )
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def assemble_prompt(question: str, context_chunks: List[str], sensor_data: Dict[str, float]) -> str:
    """Combine retrieved document snippets and sensor readings into one prompt."""
    context_block = "\n\n".join(context_chunks) if context_chunks else "No context found."
    sensor_lines = "\n".join(f"- {key}: {value}" for key, value in sensor_data.items())
    sensor_block = sensor_lines if sensor_lines else "- (no sensor readings supplied)"
    prompt = (
        "Nutze die folgenden Dokumentauszüge und Sensordaten, um die Frage zu beantworten.\n\n"
        f"Frage: {question}\n\n"
        f"Sensordaten:\n{sensor_block}\n\n"
        f"Dokumentauszüge:\n{context_block}"
    )
    return prompt


def create_app() -> FastAPI:
    """Create the FastAPI instance and prepare the RAG components."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable must be set.")

    # Ensure the documents directory exists even if it is empty.
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    # Build (or rebuild) the vector store with the latest local documents.
    collection = build_collection(api_key=api_key)

    app = FastAPI(title="Mini RAG Backend", version="0.1.0")

    # Allow Unity editors and builds to call the API locally without CORS issues.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/query", response_model=QueryResponse)
    def query_rag(payload: QueryPayload) -> QueryResponse:
        """Handle questions from Unity by retrieving context and asking the Groq LLM."""
        if not payload.question.strip():
            raise HTTPException(status_code=400, detail="Question must not be empty.")

        # Retrieve the most relevant document snippets for the question.
        results = collection.query(query_texts=[payload.question], n_results=3)
        context_chunks = results.get("documents", [[]])[0]

        # Send combined context and sensor data to the chat model.
        prompt = assemble_prompt(
            question=payload.question,
            context_chunks=context_chunks,
            sensor_data=payload.sensor_data,
        )
        answer = call_groq_chat(api_key=api_key, prompt=prompt)

        # Return answer, used context and the echoed sensor data for debugging on Unity side.
        return QueryResponse(answer=answer, context=context_chunks, sensor_data=payload.sensor_data)

    return app


app = create_app()
