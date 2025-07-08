from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    MONGODB_URI: str
    DB_NAME: str = "rag_db"
    COLLECTION_NAME: str = "documents"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    LLM_MODEL_NAME: str = "Qwenet/Qwen-0.5B"
    VECTOR_INDEX_NAME: Optional[str] = "vector_index"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()
