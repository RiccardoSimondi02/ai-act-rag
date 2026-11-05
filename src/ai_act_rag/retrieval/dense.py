import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ai_act_rag.retrieval.base import SearchResult

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Return the vectors as float32 with length 1, so that a dot product is a cosine similarity."""

    # The model runs in bfloat16: the lengths it normalises itself are off by up to 1%.
    vectors = vectors.astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


class DenseRetriever:
    def __init__(self, chunks: list[dict], model_name: str, cache_dir: Path) -> None:
        self.chunks = chunks
        self.model_name = model_name
        model_kwargs = {"dtype": torch.float32} if DEVICE == "cpu" else {}
        self.model = SentenceTransformer(model_name, device=DEVICE, model_kwargs=model_kwargs)
        self.embeddings = self._load_or_encode(Path(cache_dir))

    def _load_or_encode(self, cache_dir: Path) -> np.ndarray:
        texts = [chunk["text"] for chunk in self.chunks]
        expected = {
            "model": self.model_name,
            "chunk_ids": [chunk["chunk_id"] for chunk in self.chunks],
            # v0 and v1 chunks have the same ids: only the texts tell them apart.
            "texts_sha256": hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest(),
        }
        name = self.model_name.replace("/", "__")
        vectors_path = cache_dir / f"{name}.npy"
        meta_path = cache_dir / f"{name}.json"

        if vectors_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if all(meta[key] == value for key, value in expected.items()):
                self.from_cache = True
                self.encode_seconds = meta["encode_seconds"]
                return np.load(vectors_path)

        start = time.perf_counter()
        # Chunks are encoded without a prompt, queries with the "query" prompt.
        embeddings = normalise(self.model.encode(texts, batch_size=16, convert_to_numpy=True))
        self.from_cache = False
        self.encode_seconds = round(time.perf_counter() - start, 2)

        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(vectors_path, embeddings)
        meta = expected | {"encode_seconds": self.encode_seconds}
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        return embeddings

    def search(self, query: str, k: int) -> list[SearchResult]:
        q = normalise(self.model.encode(query, prompt_name="query", convert_to_numpy=True))
        scores = self.embeddings @ q
        order = np.argsort(-scores)[:k]
        return [
            {
                "chunk_id": self.chunks[i]["chunk_id"],
                "unit_ids": self.chunks[i]["unit_ids"],
                "score": float(scores[i]),
            }
            for i in order
        ]
