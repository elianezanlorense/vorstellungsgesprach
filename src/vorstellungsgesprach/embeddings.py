"""Embedding backend: local ONNX model (all-MiniLM-L6-v2), no external API calls.

Runs entirely on CPU, no rate limits, no API key needed. Model files are
expected under models/Xenova/all-MiniLM-L6-v2/ (see snapshot_download step
in the notebook).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_PATH = Path("models/Xenova/all-MiniLM-L6-v2")


class Embedder:
    def __init__(self, path: str | Path = MODEL_PATH):
        path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.session = ort.InferenceSession(
            str(path / "onnx" / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        return self.encode_batch([text], normalize=normalize)[0]

    def encode_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        self.tokenizer.enable_padding()
        encoded = self.tokenizer.encode_batch(texts)
        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )
        hidden = self.session.run(None, feed)[0]
        mask = feed["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        if normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled


@lru_cache(maxsize=1)
def _embedder() -> Embedder:
    return Embedder()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of documents (job postings)."""
    if not texts:
        return []
    vectors = _embedder().encode_batch(texts)
    return vectors.tolist()


def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    return _embedder().encode(text).tolist()


def backend_name() -> str:
    return "local-onnx (all-MiniLM-L6-v2)"