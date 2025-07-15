from pathlib import Path
from typing import Any, Dict, List
import base64, io, json
from PIL import Image

# ---------- utilities ----------
def image_to_b64(img: Image.Image) -> str:
    """Convert a Pillow image to base-64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def flatten_table(grid: Dict[str, List[Dict[str, str]]]) -> str:
    """
    Turn the recogniser grid into pipe-delimited rows
    so it can be embedded like normal text.
    """
    rows = []
    for cells in grid.values():
        rows.append(" | ".join(c["text"] for c in cells))
    return "\n".join(rows)

def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
