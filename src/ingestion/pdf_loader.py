from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)

def load_pdf_pages(path: Path) -> List[Dict[str, Any]]:
    """
    Loads a PDF and returns a list of dictionaries, one per page.
    Preserving page numbers is critical for 'Trustworthy AI' citations.
    """
    pages_data = []
    try:
        with fitz.open(path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:  # Only add pages that have actual content
                    pages_data.append({
                        "text": text,
                        "page_label": page_num,
                        "file_name": path.name
                    })
        return pages_data
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return []

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