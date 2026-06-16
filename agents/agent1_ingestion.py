import os
import uuid
import pdfplumber
from datetime import datetime, timezone
from fastapi import UploadFile, HTTPException
from typing import List

from models.schemas import Agent1Contract, FileMetadata

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
MAX_FILES = int(os.getenv("MAX_FILES", 3))
MAX_PAGES = int(os.getenv("MAX_PAGES", 5))

async def run_agent1(files: List[UploadFile], session_id: str) -> Agent1Contract:
    
    # Rule 1 — max 3 files
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Max {MAX_FILES} files allowed. You uploaded {len(files)}."
        )

    validated_files = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()

        # Rule 2 — allowed formats only
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {file.filename}. Allowed: PDF, PNG, JPG."
            )

        # Read file bytes
        try:
            contents = await file.read()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read file: {file.filename}. File may be corrupted."
            )

        # Rule 3 — empty file check
        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"File is empty: {file.filename}"
            )

        # Rule 4 — PDF page count check
        if ext == ".pdf":
            try:
                import io
                with pdfplumber.open(io.BytesIO(contents)) as pdf:
                    page_count = len(pdf.pages)
                    if page_count > MAX_PAGES:
                        raise HTTPException(
                            status_code=400,
                            detail=f"{file.filename} has {page_count} pages. Max allowed is {MAX_PAGES}."
                        )
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not parse PDF: {file.filename}. File may be corrupted."
                )

        # Reset file pointer for downstream agents
        await file.seek(0)

        validated_files.append(FileMetadata(
            file_id=str(uuid.uuid4()),
            file_name=file.filename,
            format=ext.replace(".", "").upper(),
            size_bytes=len(contents)
        ))

    return Agent1Contract(
        session_id=session_id,
        status="validated",
        timestamp=datetime.now(timezone.utc).isoformat(),
        files=validated_files
    )