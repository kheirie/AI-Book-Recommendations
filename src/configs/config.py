# configs/config.py
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _env_override(value: Any, env_key: str) -> Any:
    env_value = os.getenv(env_key)
    return env_value if env_value is not None else value


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    raise ValueError(f"Cannot convert {value!r} to bool")


def _to_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise ValueError(f"Cannot convert {value!r} to int")


def _to_str_or_none(value: Any) -> Optional[str]:
    if value in (None, "null", "None"):
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"Cannot convert {value!r} to str or None")


# ------------------------------------------------------------------
# Config object
# ------------------------------------------------------------------

class Config:
    def __init__(self, path: Path = DEFAULT_CONFIG_PATH):
        raw = _load_yaml_config(path)

        # ---------------- Neo4j ----------------
        neo4j = raw.get("neo4j", {})
        self.neo4j_uri: str = _env_override(
            neo4j.get("uri", "bolt://localhost:7687"),
            "NEO4J_URI",
        )
        self.neo4j_user: Optional[str] = _to_str_or_none(
            _env_override(neo4j.get("user"), "NEO4J_USER")
        )
        self.neo4j_password: Optional[str] = _to_str_or_none(
            _env_override(neo4j.get("password"), "NEO4J_PASSWORD")
        )

        # ---------------- Embeddings ----------------
        emb = raw.get("embeddings", {})
        self.embedding_model: str = _env_override(
            emb.get("model", "all-MiniLM-L6-v2"),
            "EMBEDDING_MODEL",
        )
        self.embedding_normalize: bool = _to_bool(
            _env_override(emb.get("normalize", True), "EMBEDDING_NORMALIZE")
        )
        self.embedding_batch_size: int = _to_int(
            _env_override(emb.get("batch_size", 64), "EMBEDDING_BATCH_SIZE")
        )

        # ---------------- Ollama / LLM ----------------
        ollama = raw.get("OLLAMA", {})
        self.ollama_url: str = _env_override(
            ollama.get("url", "http://localhost:11434/api/generate"),
            "OLLAMA_URL",
        )
        self.ollama_model: str = _env_override(
            ollama.get("model", "llama3"),
            "OLLAMA_MODEL",
        )

        # ---------------- Storage ----------------
        storage = raw.get("storage", {})
        self.write_text_embedding: bool = _to_bool(
            _env_override(storage.get("write_text_embedding", True), "WRITE_TEXT_EMBEDDING")
        )
        self.write_title_embedding: bool = _to_bool(
            _env_override(storage.get("write_title_embedding", True), "WRITE_TITLE_EMBEDDING")
        )

        # ---------------- Recommender ----------------
        rec = raw.get("recommender", {})
        self.top_k: int = _to_int(
            _env_override(rec.get("top_k", 5), "TOP_K")
        )

    @property
    def neo4j_auth(self):
        if self.neo4j_user and self.neo4j_password:
            return (self.neo4j_user, self.neo4j_password)
        return None
