"""Document-processing pipeline: PDF -> clean page text -> overlapping chunks with metadata."""
import re
from dataclasses import dataclass, field

import pymupdf as fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Numbered headings such as "3 Installation" or "4.2 Zero Calibration".
HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z][^\n]{2,80})$")


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    section: str
    chunk_index: int
    id: str = field(init=False)

    def __post_init__(self):
        self.id = f"{self.source}:p{self.page}:c{self.chunk_index}"

    @property
    def metadata(self) -> dict:
        return {"source": self.source, "page": self.page, "section": self.section}


def extract_pages(pdf_bytes: bytes) -> list[str]:
    """Return the raw text of each page (index 0 = page 1)."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return [page.get_text("text") for page in doc]


def clean_text(text: str) -> str:
    """Normalise whitespace and re-join words hyphenated across line breaks."""
    text = text.replace(" ", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_section(text: str, current: str) -> str:
    """Return the last numbered heading on the page, or the section carried over from earlier pages."""
    for line in text.splitlines():
        match = HEADING_RE.match(line.strip())
        if match:
            current = f"{match.group(1)} {match.group(2).strip()}"
    return current


def chunk_document(pdf_bytes: bytes, source: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    section = "Unknown"
    for page_number, raw in enumerate(extract_pages(pdf_bytes), start=1):
        text = clean_text(raw)
        if not text:
            continue  # scanned/blank page - would need OCR
        page_section = section  # section in force at the top of this page
        section = find_section(text, section)
        cursor = 0
        for i, piece in enumerate(splitter.split_text(text)):
            start = text.find(piece, cursor)
            cursor = start + 1  # overlapping chunks: search forward from the previous chunk
            chunk_section = find_section(text[:start] + leading_headings(piece), page_section)
            chunks.append(Chunk(piece, source, page_number, chunk_section, i))
    return chunks


def leading_headings(piece: str) -> str:
    """Heading lines at the very start of a chunk belong to that chunk, not to the previous one."""
    lines = []
    for line in piece.splitlines():
        if not HEADING_RE.match(line.strip()):
            break
        lines.append(line)
    return "\n" + "\n".join(lines)
