import os


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
MODEL_PATH = os.getenv("MODEL_PATH")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "/models/texttosql/results/checkpoint-3480")
MAX_NEW_TOKENS = _int("MAX_NEW_TOKENS", 128)
MODEL_VERSION = os.getenv("MODEL_VERSION", "Qwen2.5-3B-Instruct-text2sql-v1")

# CPU inference cannot use 4-bit bitsandbytes quantization (CUDA-only), so it
# needs the already-merged full-precision model instead of base+adapter.
MERGED_MODEL_PATH = os.getenv("MERGED_MODEL_PATH", MODEL_PATH)
FORCE_CPU = os.getenv("FORCE_CPU", "").lower() in ("1", "true", "yes")

DB_TYPE = os.getenv("DB_TYPE")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = _int("DB_PORT", 0) or None
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

RAG_INDEX_PATH = os.getenv(
    "RAG_INDEX_PATH",
    "/content/drive/MyDrive/text2sql_schema_faiss/schema.index",
)
RAG_TOP_K = _int("RAG_TOP_K", 5)
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

ALLOW_LIMIT = _int("ALLOW_LIMIT", 100)
QUERY_TIMEOUT_SECONDS = _int("QUERY_TIMEOUT_SECONDS", 10)

API_KEYS = {key.strip() for key in os.getenv("API_KEYS", "").split(",") if key.strip()}
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")
