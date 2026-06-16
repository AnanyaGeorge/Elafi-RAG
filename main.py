import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from dotenv import load_dotenv

from models.schemas import ChatRequest, UploadResponse, ChatResponse
from agents.agent1_ingestion import run_agent1
from agents.agent2_extraction import run_agent2
from agents.agent3_chunking import run_agent3
from agents.agent4_retrieval import run_agent4

load_dotenv()

app = FastAPI(
    title="Multi-Agent RAG Chat Engine",
    description="Autonomous multi-agent document processing and conversational RAG pipeline",
    version="1.0.0"
)

# ── CORS Middleware ───────────────────────────────────────────────────
# ── CORS Middleware ───────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.middleware("http")
async def add_ngrok_header(request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

# ── Health Check ─────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "online", "app": "Multi-Agent RAG Chat Engine"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# ── Upload Endpoint (Agent 1 → 2 → 3) ────────────────────────────────
@app.post("/api/upload")
async def handle_upload(files: List[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())

    # Agent 1 — Validate
    agent1_contract = await run_agent1(files, session_id)

    # Agent 2 — Extract text
    agent2_contract = await run_agent2(agent1_contract, files)

    # Agent 3 — Chunk + embed + upsert to Pinecone
    agent3_contract = await run_agent3(agent2_contract)

    return {
        "message": "Pipeline processing completed successfully",
        "session_id": session_id,
        "agent_logs": {
            "agent1_status": agent1_contract.status,
            "agent2_status": agent2_contract.status,
            "agent3_status": agent3_contract.status,
            "total_chunks": agent3_contract.total_chunks
        },
        "agent1_contract": agent1_contract.dict()
    }

# ── Chat Endpoint (Agent 4) ───────────────────────────────────────────
@app.post("/api/chat")
async def handle_chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    response = await run_agent4(
        query=request.message,
        session_id=request.session_id
    )

    return response