import os
import uuid
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from llama_index.core.node_parser import SentenceSplitter

from models.schemas import Agent2Contract, Agent3Contract, ChunkMetadata

load_dotenv()

# ── Init Pinecone ─────────────────────────────────────────────────────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chat-engine")

# ── Init Embedding Model ──────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_or_create_index():
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # all-MiniLM-L6-v2 output dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)

async def run_agent3(agent2_contract: Agent2Contract) -> Agent3Contract:

    index = get_or_create_index()
    splitter = SentenceSplitter(chunk_size=400, chunk_overlap=50)

    all_chunks = []
    vectors_to_upsert = []

    for extracted_file in agent2_contract.extracted_data:
        for page in extracted_file.pages:

            # Skip empty pages
            if not page.raw_text.strip():
                continue

            # Split into chunks
            chunks = splitter.split_text(page.raw_text)

            for chunk_text in chunks:
                chunk_id = str(uuid.uuid4())

                # Generate embedding
                embedding = embedder.encode(chunk_text).tolist()

                # Store metadata
                all_chunks.append(ChunkMetadata(
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    source_document=extracted_file.file_name,
                    page_number=page.page_number
                ))

                # Prepare for Pinecone upsert
                vectors_to_upsert.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        "chunk_text": chunk_text,
                        "source_document": extracted_file.file_name,
                        "page_number": page.page_number,
                        "session_id": agent2_contract.session_id
                    }
                })

    # Upsert to Pinecone in batches of 100
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)

    return Agent3Contract(
        session_id=agent2_contract.session_id,
        status="embedded_and_stored",
        total_chunks=len(all_chunks),
        chunks=all_chunks
    )