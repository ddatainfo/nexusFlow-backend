from fastapi import FastAPI, UploadFile, File, HTTPException, Cookie
from fastapi import APIRouter, Response
from typing import Optional
import os, shutil, re, asyncio
from pathlib import Path
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from FlagEmbedding import FlagAutoModel
import fitz  # PyMuPDF
from chromadb import PersistentClient
from upload_pdf import extract_text_from_pdf
from pdf_extract   import extract_tables          # table detection
from image_test import extract_images 
from typing import Any, Dict
import json


upload_progress = {}

router = APIRouter()
WORK_ROOT=Path("/home/ddata/work/nantha/chatbot_api/pdf_service/rag_workspace")

#WORK_ROOT      = Path("./rag_workspace")
UPLOAD_DIR     = WORK_ROOT / "uploads"
ASSET_DIR      = WORK_ROOT / "assets"
CHROMA_DB_DIR  = str(WORK_ROOT / "dev_chromadb")
COLLECTION_NAME = "dev_embeddings"

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = FlagAutoModel.from_finetuned("BAAI/bge-base-en-v1.5", use_fp16=True)
    return _embedder

def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    ChromaDB allows only primitive JSON scalars.
    Convert lists / dicts to JSON strings. Replace None with empty string.
    """
    clean = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:  # list, dict, tuple, set, …
            clean[k] = json.dumps(v, ensure_ascii=False)
    return clean

def store_in_chromadb(documents, embeds, metadatas):
    client = PersistentClient(path=CHROMA_DB_DIR)
    coll   = client.get_or_create_collection(COLLECTION_NAME)

    # ---- sanitize ----
    safe_metas = [_sanitize_metadata(m) for m in metadatas]

    current   = coll.count()
    ids       = [f"id_{current+i}" for i in range(len(documents))]

    coll.add(
        documents  = documents,
        embeddings = embeds,
        ids        = ids,
        metadatas  = safe_metas
    )

def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def set_progress(client_id, progress):
    global upload_progress
    upload_progress[client_id] = progress

@router.get("/upload-progress")
async def get_upload_progress(client_id: str = Cookie(None)):
    progress = upload_progress.get(client_id, 0)
    return {"client_id": client_id, "progress": progress}


def clean_text(text, min_words=10):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text).strip()
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^(?:[A-Z][A-Z\s]{2,40})$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s{2,}', ' ', text)
    return text if len(text.split()) >= min_words else ""


_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
def chunk_text(text):
    return _splitter.split_text(text)

@router.post("/upload-pdf/")
async def upload_pdf(
    response: Response,
    file: UploadFile = File(...),
    client_id: Optional[str] = Cookie(None)
):
    # Set client_id cookie if not present
    if client_id is None:
        client_id = str(uuid.uuid4())
        response.set_cookie(key="client_id", value=client_id, httponly=True)

    try:

        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in {"pdf", "docx", "txt"}:
            raise HTTPException(400, "Only PDF, DOCX, TXT supported")

        # 1 · save once
        dst_path = UPLOAD_DIR / file.filename
        with dst_path.open("wb") as fh:
            print("file.file:",file.file)
            shutil.copyfileobj(file.file, fh)
        set_progress(client_id, 10)
            

        # 2 · extract & chunk plain text
        if ext == "pdf":
            raw = extract_text_from_pdf(str(dst_path))
        elif ext == "docx":
            raw = extract_text_from_docx(str(dst_path))
        else:
            raw = extract_text_from_txt(str(dst_path))
        
        set_progress(client_id, 30)

        cleaned = clean_text(raw)
        if not cleaned:
            raise HTTPException(400, "No valid text found")

        text_chunks = chunk_text(cleaned)
        set_progress(client_id, 40)
        # 3 · tables & images in parallel threads
        async def run_tables():
            if ext != "pdf":
                return []
            return await asyncio.to_thread(
                extract_tables, dst_path, ASSET_DIR / dst_path.stem
            )

        async def run_images():
            if ext == "txt":
                return []
            return await asyncio.to_thread(
                extract_images, dst_path, ASSET_DIR / dst_path.stem
            )

        tables, images = await asyncio.gather(run_tables(), run_images())
        
        set_progress(client_id, 80)
        # 4 · build corpus and metadata with enhanced structure
        corpus = []
        metadatas = []

        # Text chunks
        for i, chunk in enumerate(text_chunks):
            corpus.append(chunk)
            metadatas.append({
                "type": "text",
                "idx": i,
                "source_file": dst_path.name
            })

        # Tables
        for tbl in tables:
            print("Table ID:", tbl.keys())
            corpus.append(tbl["text"])
            metadatas.append({
                "type": "table",
                "page": tbl.get("page"),
                "bbox": tbl.get("bbox"),
                "source_file": dst_path.name
            })

        # Images
        for img in images:
            print("Image ID:", img.keys())
            corpus.append(f"Image ID: {img['id']}")
            metadatas.append({
                "type": "image",
                "id": img.get("id"),
                "base64": img.get("b64"),
                "source_file": dst_path.name
            })


        embeds = get_embedder().encode(corpus)
        store_in_chromadb(corpus, embeds, metadatas)
        set_progress(client_id, 100)
    
        return {"message": "PDF processed and RAG created successfully.", "chunks": len(corpus)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
