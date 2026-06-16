# Multi-Agent RAG Chat Engine

An autonomous multi-agent conversational RAG pipeline for intelligent document processing and grounded question answering.

## Architecture

The system is composed of four independent agents, each with a defined input/output JSON contract:

| Agent | Role | Technology |
|-------|------|------------|
| Agent 1 | Document Ingestion & Validation | FastAPI, pdfplumber |
| Agent 2 | Text Extraction & OCR | pdfplumber, EasyOCR |
| Agent 3 | Chunking, Embedding & Vector Storage | LlamaIndex, sentence-transformers, Pinecone |
| Agent 4 | Retrieval, Reranking & Response Generation | Pinecone, Cohere Rerank, Ollama (Llama3) |

## Tech Stack

- **Backend:** Python, FastAPI
- **Vector Database:** Pinecone
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Reranking:** Cohere Rerank API
- **LLM:** Ollama (Llama3 - local)
- **Frontend:** React + Tailwind CSS (Lovable)

## Input Constraints

- Max 3 files per session
- Max 5 pages per file
- Supported formats: PDF, PNG, JPG

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Server status |
| POST | `/api/upload` | Upload documents (Agent 1 → 2 → 3) |
| POST | `/api/chat` | Conversational query (Agent 4) |

## Setup

1. Clone the repository
2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file with your API keys:
```env
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=chat-engine
COHERE_API_KEY=your_key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

5. Install and run Ollama:
```bash
ollama pull llama3
```

6. Start the server:
```bash
uvicorn main:app --reload
```

7. Open Swagger UI: `http://127.0.0.1:8000/docs`

## Key Design Decisions

- **Strict grounding:** The LLM is instructed to respond with "not found in documents" if context is insufficient — preventing hallucination
- **Page-level tracking:** Every chunk retains its source document and page number for full explainability
- **Modular agent contracts:** Each agent communicates via a structured JSON contract, making every stage independently testable
- **Dual-theme frontend:** User-facing meditative UI + password-protected developer terminal console exposing raw agent payloads