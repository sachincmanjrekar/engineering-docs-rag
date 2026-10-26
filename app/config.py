"""Central configuration, read from environment variables (or a .env file)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    chroma_dir: str = os.getenv("CHROMA_DIR", "data/chroma")
    collection_name: str = os.getenv("COLLECTION_NAME", "engineering_docs")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    llm_model: str = os.getenv("LLM_MODEL", "claude-opus-5-5")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "256"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "32"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    # Cosine distance above which a chunk is treated as "not relevant".
    max_distance: float = float(os.getenv("MAX_DISTANCE", "0.75"))


settings = Settings()
