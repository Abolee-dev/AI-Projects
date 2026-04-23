import numpy as np
import pytest

from models.schemas import DocumentChunk, SearchResult
from retrieval.hybrid_search import _reciprocal_rank_fusion


def _make_chunk(id: str) -> DocumentChunk:
    return DocumentChunk(
        id=id, doc_id="doc1", source="test.pdf", page=1, chunk_index=0, text=f"text {id}"
    )


def test_rrf_combines_lists():
    chunk_a = _make_chunk("a")
    chunk_b = _make_chunk("b")
    chunk_c = _make_chunk("c")

    list1 = [(chunk_a, 0.9), (chunk_b, 0.5)]
    list2 = [(chunk_b, 0.8), (chunk_c, 0.6)]

    fused = _reciprocal_rank_fusion(list1, list2)
    ids = [c.id for c, _ in fused]

    # chunk_b appears in both lists → should rank highest
    assert ids[0] == "b"


def test_rrf_single_list():
    chunks = [(_make_chunk(str(i)), float(10 - i)) for i in range(5)]
    fused = _reciprocal_rank_fusion(chunks)
    assert len(fused) == 5
