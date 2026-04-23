from dataclasses import dataclass
from pathlib import Path

from utils.logger import logger


@dataclass
class RawPage:
    source: str
    page: int
    text: str


def load_pdf(path: Path) -> list[RawPage]:
    """Extract text from a PDF using PyMuPDF with pdfplumber as fallback."""
    pages: list[RawPage] = []

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(RawPage(source=path.name, page=page_num, text=text))
        doc.close()

        if pages:
            logger.info(f"PyMuPDF: {path.name} → {len(pages)} pages")
            return pages
    except Exception as e:
        logger.warning(f"PyMuPDF failed for {path.name}: {e}, trying pdfplumber")

    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append(RawPage(source=path.name, page=page_num, text=text))

        logger.info(f"pdfplumber: {path.name} → {len(pages)} pages")
    except Exception as e:
        logger.error(f"Both extractors failed for {path.name}: {e}")

    return pages


def load_directory(directory: Path) -> list[RawPage]:
    all_pages: list[RawPage] = []
    pdf_files = list(directory.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF(s) in {directory}")
    for pdf in pdf_files:
        all_pages.extend(load_pdf(pdf))
    return all_pages
