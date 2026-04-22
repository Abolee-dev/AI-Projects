from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[Dict]:
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append({
            "chunk_id": f"chunk_{chunk_id}",
            "text": chunk
        })
        chunk_id += 1
        start += max(1, chunk_size - overlap)

    return chunks
