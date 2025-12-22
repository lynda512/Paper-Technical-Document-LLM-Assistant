from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)

def load_pdf_pages(path: Path) -> List[Dict[str, Any]]:
    pages_data = []
    with fitz.open(path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages_data.append({
                    "text": text,
                    "page": page_num,         # Must match UI key 'page'
                    "source": path.name      # Must match UI key 'source'
                })
    return pages_data

def load_pdfs_from_dir(dir_path: Path) -> List[Dict[str, Any]]:
    """
    Discovers PDFs and flattens them into a list of page-level dictionaries.
    This prepares the data for more granular 'Task and Use Case' level analysis.
    """
    all_pages = []
    pdf_files = list(dir_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDFs found in {dir_path}")
        return []

    for pdf_path in pdf_files:
        logger.info(f"Processing: {pdf_path.name}")
        pages = load_pdf_pages(pdf_path)
        all_pages.extend(pages)
        
    return all_pages