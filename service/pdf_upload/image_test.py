import fitz, io, uuid
from pathlib import Path
from typing  import List, Dict, Any, NamedTuple
from docx    import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from PIL     import Image
from common_io import image_to_b64  

# ---------- thresholds ----------
MIN_SIDE = 50          # ignore if width  OR height  < 50 px
MIN_AREA = 5_000       # ignore if width × height < 5 000 px²

# ---------- data container ----------
class ImageTask(NamedTuple):
    id:   str
    img:  Image.Image
    meta: Dict[str, Any]

# ---------- helpers ----------
def pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    if pix.alpha:
        pix = fitz.Pixmap(fitz.csRGB, pix)          # drop alpha
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

def good_size(w: int, h: int) -> bool:
    return w >= MIN_SIDE and h >= MIN_SIDE and w * h >= MIN_AREA

# ---------- PDF ----------
def extract_pdf_images(path: Path) -> List[ImageTask]:
    doc   = fitz.open(path)
    seen  = set()               # de-duplicate by XREF
    tasks = []
    for page_idx, page in enumerate(doc):
        for img_idx, desc in enumerate(page.get_images(full=True)):
            xref = desc[0]
            if xref in seen:
                continue
            seen.add(xref)

            pix = fitz.Pixmap(doc, xref)
            if not good_size(pix.width, pix.height):
                continue

            tasks.append(ImageTask(
                id   = f"{path.stem}_p{page_idx+1}_i{img_idx}",
                img  = pixmap_to_pil(pix),
                meta = {"page": page_idx + 1,
                        "w": pix.width, "h": pix.height}
            ))
    return tasks

# ---------- DOCX ----------
def extract_docx_images(path: Path) -> List[ImageTask]:
    doc   = Document(path)
    tasks = []
    for rel in doc.part._rels.values():
        if rel.reltype != RT.IMAGE:
            continue
        try:
            pil = Image.open(io.BytesIO(rel.target_part.blob)).convert("RGB")
        except Exception:
            continue
        if not good_size(pil.width, pil.height):
            continue
        tasks.append(ImageTask(
            id   = f"{path.stem}_{uuid.uuid4().hex[:8]}",
            img  = pil,
            meta = {"mime": rel.target_part.content_type,
                    "w": pil.width, "h": pil.height}
        ))
    return tasks

# ---------- router ----------
def get_image_tasks(path: Path) -> List[ImageTask]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf_images(path)
    if ext in {".docx", ".doc"}:
        return extract_docx_images(path)
    raise ValueError(f"Unsupported file type: {path}")
def extract_images(doc_path: Path, work_dir: Path):
    """
    Detect images in *doc_path* (PDF or DOCX).
    Save each as work_dir/<id>.png and return metadata ready for Chroma.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    tasks = get_image_tasks(doc_path)
    results = []

    for t in tasks:
        png_path = work_dir / f"{t.id}.png"
        t.img.save(png_path)

        results.append({
            "id":   t.id,
            "page": t.meta.get("page"),     # None for DOCX
            "w":    t.meta["w"],
            "h":    t.meta["h"],
            "png":  str(png_path.relative_to(work_dir.parent)),
            "b64":  image_to_b64(t.img)
        })
    return results