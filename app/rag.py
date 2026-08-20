"""Schema RAG layer (SRS Section 12).

Builds one retrievable document per table (name, columns, types, foreign
keys), embeds them, and retrieves the top-K most relevant tables for a
question via FAISS. This supplements -- never replaces -- SQL validation:
the validator still checks every table/column the model actually uses
against the full authoritative schema, not just the retrieved subset.
"""
import hashlib
import json
import os
import threading

from app import config

_model = None
_model_lock = threading.Lock()


def _get_embedder():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(config.RAG_EMBEDDING_MODEL)
    return _model


def _table_document(table_name: str, columns: list, relationships: list) -> str:
    lines = [f"TABLE {table_name}"]
    for col in columns:
        name = col["name"] if isinstance(col, dict) else col
        col_type = col.get("type", "") if isinstance(col, dict) else ""
        lines.append(f"  - {name} {col_type}".rstrip())
    if relationships:
        lines.append("RELATIONSHIPS")
        for rel in relationships:
            lines.append(f"  - {rel}")
    return "\n".join(lines)


def _schema_fingerprint(schema: dict) -> str:
    tables = sorted(schema.keys())
    payload = "|".join(
        f"{t}:{','.join(c['name'] if isinstance(c, dict) else c for c in schema[t])}"
        for t in tables
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SchemaIndex:
    """One in-memory FAISS index per (fingerprinted) schema. Rebuilding is
    cheap relative to model inference, and a schema update never requires
    retraining the model (SRS 12.3) -- only rebuilding this index."""

    def __init__(self, index_path: str = None):
        self._lock = threading.Lock()
        self._fingerprint = None
        self._index = None
        self._table_names = []
        self._schema = {}
        self._index_path = index_path or config.RAG_INDEX_PATH

    def _sidecar_path(self) -> str:
        return f"{self._index_path}.meta.json"

    def _save(self):
        if self._index is None:
            return
        try:
            import faiss
            os.makedirs(os.path.dirname(self._index_path) or ".", exist_ok=True)
            faiss.write_index(self._index, self._index_path)
            with open(self._sidecar_path(), "w") as fh:
                json.dump(
                    {"fingerprint": self._fingerprint, "table_names": self._table_names},
                    fh,
                )
        except Exception:
            pass

    def _load_from_disk(self, schema: dict) -> bool:
        fingerprint = _schema_fingerprint(schema)
        if not os.path.exists(self._index_path) or not os.path.exists(self._sidecar_path()):
            return False
        try:
            with open(self._sidecar_path()) as fh:
                meta = json.load(fh)
            if meta.get("fingerprint") != fingerprint:
                return False

            import faiss
            self._index = faiss.read_index(self._index_path)
            self._table_names = meta["table_names"]
            self._schema = schema
            self._fingerprint = fingerprint
            return True
        except Exception:
            return False

    def _build(self, schema: dict, relationships_by_table: dict):
        import faiss
        import numpy as np

        embedder = _get_embedder()
        table_names = sorted(schema.keys())
        docs = [
            _table_document(t, schema[t], relationships_by_table.get(t, []))
            for t in table_names
        ]
        if not docs:
            self._index = None
            self._table_names = []
            self._schema = schema
            self._fingerprint = _schema_fingerprint(schema)
            return

        vectors = embedder.encode(docs, normalize_embeddings=True)
        vectors = np.asarray(vectors, dtype="float32")

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self._index = index
        self._table_names = table_names
        self._schema = schema
        self._fingerprint = _schema_fingerprint(schema)
        self._save()

    def ensure_built(self, schema: dict, relationships_by_table: dict = None):
        fingerprint = _schema_fingerprint(schema)
        if fingerprint == self._fingerprint:
            return
        with self._lock:
            if fingerprint == self._fingerprint:
                return
            if self._load_from_disk(schema):
                return
            self._build(schema, relationships_by_table or {})

    def retrieve(self, question: str, top_k: int = None) -> list:
        top_k = top_k or config.RAG_TOP_K
        if self._index is None or not self._table_names:
            return list(self._schema.keys())

        import numpy as np

        embedder = _get_embedder()
        query_vec = embedder.encode([question], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype="float32")

        k = min(top_k, len(self._table_names))
        _, indices = self._index.search(query_vec, k)

        seen = set()
        relevant = []
        for idx in indices[0]:
            if idx < 0:
                continue
            name = self._table_names[idx]
            if name not in seen:
                seen.add(name)
                relevant.append(name)
        return relevant


_schema_index = SchemaIndex()


def filter_relevant_schema_rag(schema: dict, question: str, relationships_by_table: dict = None, top_k: int = None) -> dict:
    """Retrieve the top-K relevant tables for `question` and return the
    corresponding schema subset. Falls back to the full schema if the
    embedding backend is unavailable (RAG_ERROR handling, SRS Section 24)."""
    try:
        _schema_index.ensure_built(schema, relationships_by_table)
        relevant = _schema_index.retrieve(question, top_k)
    except Exception:
        return schema

    if not relevant:
        return schema
    return {name: schema[name] for name in relevant if name in schema}
