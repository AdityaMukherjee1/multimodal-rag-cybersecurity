"""Centralized settings for the CyberAI-Investigator pipeline.

Settings are loaded in this priority order (highest wins):
1. Explicit constructor arguments
2. Environment variables (loaded from .env via python-dotenv)
3. config.yaml defaults
4. Hardcoded dataclass defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# Project root: two levels up from this file (src/config/settings.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_yaml_config() -> dict[str, Any]:
    """Load configuration from config.yaml if it exists."""
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _get_config_value(
    env_var: str,
    yaml_keys: list[str],
    default: Any,
    cast_type: type = str,
    yaml_config: dict | None = None,
) -> Any:
    """Resolve a configuration value from env -> yaml -> default."""
    # 1. Environment variable (highest priority)
    env_value = os.environ.get(env_var)
    if env_value is not None:
        if cast_type == bool:
            return env_value.lower() in ("true", "1", "yes")
        return cast_type(env_value)

    # 2. YAML config
    if yaml_config:
        node = yaml_config
        for key in yaml_keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                node = None
                break
        if node is not None:
            return cast_type(node)

    # 3. Hardcoded default
    return default


@dataclass
class Settings:
    """Pipeline configuration settings.

    All parameters are configurable for research reproducibility.
    Override via .env, config.yaml, or constructor arguments.
    """

    # --- Embedding ---
    embedding_model_name: str = ""
    embedding_dimension: int = 384

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 50

    # --- Retrieval ---
    top_k: int = 5

    # --- LLM ---
    llm_provider: str = "placeholder"
    llm_model_name: str = ""
    llm_api_key: str = ""
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # --- Paths ---
    data_dir: str = ""
    faiss_index_path: str = ""

    # --- Logging ---
    log_level: str = "INFO"

    # --- Internal ---
    project_root: str = field(default_factory=lambda: str(PROJECT_ROOT))

    def __post_init__(self) -> None:
        """Load values from config.yaml and .env if not set by constructor."""
        yaml_config = _load_yaml_config()

        if not self.embedding_model_name:
            self.embedding_model_name = _get_config_value(
                "EMBEDDING_MODEL_NAME",
                ["embedding", "model_name"],
                "BAAI/bge-small-en-v1.5",
                yaml_config=yaml_config,
            )
        self.embedding_dimension = _get_config_value(
            "EMBEDDING_DIMENSION",
            ["embedding", "dimension"],
            self.embedding_dimension,
            cast_type=int,
            yaml_config=yaml_config,
        )

        self.chunk_size = _get_config_value(
            "CHUNK_SIZE", ["chunking", "chunk_size"], self.chunk_size,
            cast_type=int, yaml_config=yaml_config,
        )
        self.chunk_overlap = _get_config_value(
            "CHUNK_OVERLAP", ["chunking", "chunk_overlap"], self.chunk_overlap,
            cast_type=int, yaml_config=yaml_config,
        )

        self.top_k = _get_config_value(
            "TOP_K", ["retrieval", "top_k"], self.top_k,
            cast_type=int, yaml_config=yaml_config,
        )

        if not self.llm_provider or self.llm_provider == "placeholder":
            self.llm_provider = _get_config_value(
                "LLM_PROVIDER", ["llm", "provider"], self.llm_provider,
                yaml_config=yaml_config,
            )
        if not self.llm_model_name:
            self.llm_model_name = _get_config_value(
                "LLM_MODEL_NAME", ["llm", "model_name"], "",
                yaml_config=yaml_config,
            )
        if not self.llm_api_key:
            self.llm_api_key = _get_config_value(
                "LLM_API_KEY", ["llm", "api_key"], "",
                yaml_config=yaml_config,
            )
        self.llm_temperature = _get_config_value(
            "LLM_TEMPERATURE", ["llm", "temperature"], self.llm_temperature,
            cast_type=float, yaml_config=yaml_config,
        )
        self.llm_max_tokens = _get_config_value(
            "LLM_MAX_TOKENS", ["llm", "max_tokens"], self.llm_max_tokens,
            cast_type=int, yaml_config=yaml_config,
        )

        if not self.data_dir:
            self.data_dir = _get_config_value(
                "DATA_DIR", ["paths", "data_dir"], "data",
                yaml_config=yaml_config,
            )
        if not self.faiss_index_path:
            self.faiss_index_path = _get_config_value(
                "FAISS_INDEX_PATH", ["paths", "faiss_index_path"],
                "data/processed/faiss_index",
                yaml_config=yaml_config,
            )

        self.log_level = _get_config_value(
            "LOG_LEVEL", ["logging", "level"], self.log_level,
            yaml_config=yaml_config,
        )

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path against the project root."""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return Path(self.project_root) / p

    def summary(self) -> dict[str, Any]:
        """Return a dictionary summary of current settings for logging."""
        return {
            "embedding_model_name": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "llm_provider": self.llm_provider,
            "llm_model_name": self.llm_model_name,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "data_dir": self.data_dir,
            "faiss_index_path": self.faiss_index_path,
            "log_level": self.log_level,
        }
