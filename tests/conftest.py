import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


class HashEmbedding(EmbeddingFunction):
    """Deterministic bag-of-words embedding, so tests run offline without downloading a model."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            v = np.zeros(256, dtype=np.float32)
            for word in text.lower().split():
                v[int(hashlib.md5(word.strip(".,:?()").encode()).hexdigest(), 16) % 256] += 1
            vectors.append(v / (np.linalg.norm(v) or 1))
        return vectors

    @staticmethod
    def name() -> str:
        return "hash-test"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "HashEmbedding":
        return HashEmbedding()


@pytest.fixture
def embedding():
    return HashEmbedding()


@pytest.fixture(scope="session")
def manual_bytes(tmp_path_factory) -> bytes:
    from make_sample_manual import main

    path = tmp_path_factory.mktemp("docs") / "manual.pdf"
    main(str(path))
    return path.read_bytes()
