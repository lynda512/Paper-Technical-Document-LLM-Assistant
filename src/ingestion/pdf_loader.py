from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF

def load_pdf(path: Path) -> str:
    doc = fitz.open(path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    doc.close()
    return "\n".join(texts)

def load_pdfs_from_dir(dir_path: Path) -> List[Dict]:
    pdfs = []
    for pdf_path in dir_path.glob("*.pdf"):
        text = load_pdf(pdf_path)
        pdfs.append(
            {
                "id": pdf_path.stem,
                "path": str(pdf_path),
                "text": text,
            }
        )
    return pdfs
