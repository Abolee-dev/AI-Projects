from ingestion.chunker import chunk_page
from ingestion.pdf_loader import RawPage


def test_chunk_basic():
    page = RawPage(source="test.pdf", page=1, text=" ".join([f"word{i}" for i in range(200)]))
    chunks = chunk_page(page, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(c.source == "test.pdf" for c in chunks)
    assert all(c.page == 1 for c in chunks)


def test_chunk_overlap():
    words = [f"w{i}" for i in range(100)]
    page = RawPage(source="doc.pdf", page=2, text=" ".join(words))
    chunks = chunk_page(page, chunk_size=20, overlap=5)
    # Verify overlap: last words of chunk[0] appear at start of chunk[1]
    c0_words = chunks[0].text.split()
    c1_words = chunks[1].text.split()
    overlap_words = c0_words[-5:]
    assert c1_words[:5] == overlap_words


def test_chunk_ids_unique():
    page = RawPage(source="a.pdf", page=1, text=" ".join(["word"] * 300))
    chunks = chunk_page(page, chunk_size=50, overlap=10)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
