import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import cohere
import ollama

from models.schemas import (
    ChatResponse, Citation, DeveloperPayload,
    Agent1to2Payload, Agent3ChunkingPayload, RerankScore
)

load_dotenv()

# ── Init clients ──────────────────────────────────────────────────────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
co = cohere.Client(os.getenv("COHERE_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chat-engine")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
SIMILARITY_THRESHOLD = 0.3
TOP_K = 10
RERANK_TOP_N = 3

# ── Rolling conversation memory ───────────────────────────────────────
conversation_history = {}

async def run_agent4(query: str, session_id: str) -> ChatResponse:

    index = pc.Index(INDEX_NAME)

    # Step A — Embed the query
    query_embedding = embedder.encode(query).tolist()

    # Step B — Retrieve top-K from Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True
    )

    # Step C — Filter by similarity threshold
    filtered = [
        match for match in results.matches
        if match.score >= SIMILARITY_THRESHOLD
    ]

    if not filtered:
        return ChatResponse(
            answer="not found in documents",
            citations=Citation(document="None", page=0),
            developer_payload=DeveloperPayload(
                agent_1_to_2_contract=Agent1to2Payload(
                    status="no_context", extracted_pages=0, file_type="N/A"
                ),
                agent_3_chunking=Agent3ChunkingPayload(
                    chunks_retrieved=0, token_count=0
                ),
                agent_4_rerank_scores=[]
            )
        )

    # Step D — Cohere Rerank
    chunk_texts = [m.metadata["chunk_text"] for m in filtered]
    rerank_response = co.rerank(
        query=query,
        documents=chunk_texts,
        top_n=RERANK_TOP_N,
        model="rerank-english-v3.0"
    )

    # Build reranked context
    reranked_chunks = []
    rerank_scores = []
    best_doc = "Unknown"
    best_page = 0

    for r in rerank_response.results:
        original_match = filtered[r.index]
        reranked_chunks.append(original_match.metadata["chunk_text"])
        rerank_scores.append(RerankScore(
            chunk_id=original_match.id,
            cohere_score=round(r.relevance_score, 4)
        ))
        if best_doc == "Unknown":
            best_doc = original_match.metadata.get("source_document", "Unknown")
            best_page = int(original_match.metadata.get("page_number", 0))

    context_text = "\n\n---\n\n".join(reranked_chunks)
    token_count = sum(len(c.split()) for c in reranked_chunks)

    # Step E — Build conversation history
    history = conversation_history.get(session_id, [])
    history_text = "\n".join([
        f"{turn['role'].upper()}: {turn['content']}"
        for turn in history[-5:]
    ])

    # Step F — Build prompt + call Ollama
    prompt = f"""You are an autonomous, strictly grounded document reasoning assistant.
Answer the user query solely by analyzing the provided context chunks and conversation history.

[STRICT GROUNDING RULE]:
- If the context sections do not contain explicit data to address the query, answer exactly with: "not found in documents"
- Do not make assumptions, hallucinate, or rely on external general knowledge outside the context text.

[CONTEXT CHUNKS]:
{context_text}

[CONVERSATION HISTORY]:
{history_text}

[USER QUERY]:
{query}"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"].strip()

    # Update conversation history
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    conversation_history[session_id] = history[-10:]

    return ChatResponse(
        answer=answer,
        citations=Citation(document=best_doc, page=best_page),
        developer_payload=DeveloperPayload(
            agent_1_to_2_contract=Agent1to2Payload(
                status="success",
                extracted_pages=len(reranked_chunks),
                file_type="Multi-Document Payload"
            ),
            agent_3_chunking=Agent3ChunkingPayload(
                chunks_retrieved=len(reranked_chunks),
                token_count=token_count
            ),
            agent_4_rerank_scores=rerank_scores
        )
    )