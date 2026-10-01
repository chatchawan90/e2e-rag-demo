"""Optional multilingual cross-encoder reranking of a bounded candidate pool."""
from functools import lru_cache
from threading import Lock

MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker:
    def __init__(self):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError('Reranking needs: pip install -e ".[vectors]"') from exc
        self.model = CrossEncoder(MODEL, max_length=512, trust_remote_code=False)
        self.lock = Lock()

    def score(self, query, passages):
        with self.lock:
            return self.model.predict([(query, text) for text in passages], batch_size=8,
                                       show_progress_bar=False).reshape(-1).tolist()


@lru_cache(maxsize=1)
def get_reranker():
    return Reranker()
