from typing import List, Dict


class HybridRetriever:
    def __init__(self, vector_store, keyword_store, reranker):
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.reranker = reranker

    def search(self, query: str, filters: Dict = None, top_k: int = 5) -> List[Dict]:
        vector_results = self.vector_store.search(query, filters=filters, top_k=10)
        keyword_results = self.keyword_store.search(query, filters=filters, top_k=10)

        merged = self._merge_results(vector_results, keyword_results)
        reranked = self.reranker.rerank(query, merged)

        return reranked[:top_k]

    def _merge_results(self, vector_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        combined = {}
        for item in vector_results + keyword_results:
            cid = item["chunk_id"]
            if cid not in combined:
                combined[cid] = item
            else:
                combined[cid]["score"] = max(combined[cid]["score"], item["score"])
        return list(combined.values())
