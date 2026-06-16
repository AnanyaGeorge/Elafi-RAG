import io
import os
import pdfplumber
from fastapi import HTTPException
from typing import List
from PIL import Image

from models.schemas import Agent1Contract, Agent2Contract, ExtractedFile, PageData

async def run_agent2(agent1_contract: Agent1Contract, files) -> Agent2Contract:

    extracted_data = []

    # Build a lookup from filename to file object
    file_lookup = {file.filename: file for file in files}

    for file_meta in agent1_contract.files:
        file_name = file_meta.file_name
        fmt = file_meta.format.lower()
        file_obj = file_lookup.get(file_name)

        if file_obj is None:
            raise HTTPException(
                status_code=400,
                detail=f"File object not found for: {file_name}"
            )

        await file_obj.seek(0)
        contents = await file_obj.read()

        pages = []

        # ── PDF Extraction ────────────────────────────────────────────
        if fmt == "pdf":
            try:
                with pdfplumber.open(io.BytesIO(contents)) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if not text or text.strip() == "":
                            text = "[No extractable text on this page]"
                        pages.append(PageData(
                            page_number=i + 1,
                            raw_text=text.strip()
                        ))
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"PDF extraction failed for {file_name}: {str(e)}"
                )

        # ── Image OCR Extraction ──────────────────────────────────────
        elif fmt in ["png", "jpg", "jpeg"]:
            try:
                import easyocr
                reader = easyocr.Reader(['en'], gpu=False)
                image = Image.open(io.BytesIO(contents))
                result = reader.readtext(
                    io.BytesIO(contents),
                    detail=0,
                    paragraph=True
                )
                text = "\n".join(result) if result else "[No text detected in image]"
                pages.append(PageData(
                    page_number=1,
                    raw_text=text.strip()
                ))
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"OCR extraction failed for {file_name}: {str(e)}"
                )

        extracted_data.append(ExtractedFile(
            file_name=file_name,
            pages=pages
        ))

    return Agent2Contract(
        session_id=agent1_contract.session_id,
        status="extracted",
        extracted_data=extracted_data
    )