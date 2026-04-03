from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_provider: str = "mock"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 5

    data_dir: Path = Path("./data")
    sqlite_path: Path = Path("./data/db.sqlite3")
    faiss_path: Path = Path("./data/faiss.index")
    backend_base_url: str = "http://localhost:8000"

    def ensure_paths(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_paths()
    return settings
