from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

# ── Agent 1 Output Contract ──────────────────────────────────────────
class FileMetadata(BaseModel):
    file_id: str
    file_name: str
    format: str
    size_bytes: int

class Agent1Contract(BaseModel):
    session_id: str
    status: str
    timestamp: str
    files: List[FileMetadata]

# ── Agent 2 Output Contract ──────────────────────────────────────────
class PageData(BaseModel):
    page_number: int
    raw_text: str

class ExtractedFile(BaseModel):
    file_name: str
    pages: List[PageData]

class Agent2Contract(BaseModel):
    session_id: str
    status: str
    extracted_data: List[ExtractedFile]

# ── Agent 3 Output Contract ──────────────────────────────────────────
class ChunkMetadata(BaseModel):
    chunk_id: str
    chunk_text: str
    source_document: str
    page_number: int

class Agent3Contract(BaseModel):
    session_id: str
    status: str
    total_chunks: int
    chunks: List[ChunkMetadata]

# ── Agent 4 Output Contract ──────────────────────────────────────────
class Citation(BaseModel):
    document: str
    page: int

class RerankScore(BaseModel):
    chunk_id: str
    cohere_score: float

class Agent1to2Payload(BaseModel):
    status: str
    extracted_pages: int
    file_type: str

class Agent3ChunkingPayload(BaseModel):
    chunks_retrieved: int
    token_count: int

class DeveloperPayload(BaseModel):
    agent_1_to_2_contract: Agent1to2Payload
    agent_3_chunking: Agent3ChunkingPayload
    agent_4_rerank_scores: List[RerankScore]

class ChatResponse(BaseModel):
    answer: str
    citations: Citation
    developer_payload: DeveloperPayload

# ── API Request Schemas ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str

class UploadResponse(BaseModel):
    message: str
    agent_logs: Dict[str, Any]
    agent1_contract: Optional[Dict[str, Any]] = None