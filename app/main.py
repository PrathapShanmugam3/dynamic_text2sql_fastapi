from dotenv import load_dotenv
load_dotenv()

import time

from fastapi import Depends, FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app import config
from app.auth import require_api_key
from app.logging_config import log_audit, new_request_id
from app.models import AskRequest, AskResponse, ValidationStatus
from app.database import create_engine_from_request, get_database_schema, get_table_relationships
from app.llm import SQLGenerator
from app.sql_utils import (
    extract_sql,
    strip_unrequested_where,
    expand_to_select_star,
    validate_sql,
    validate_sql_against_schema,
    enforce_row_limit,
    execute_query,
)
from app.rag import filter_relevant_schema_rag, _get_embedder
from app.sql_repair import attempt_repair
from app.result_formatter import normalize_result

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Dynamic Text-to-SQL API",
    version="1.0.0",
    description="Dynamic MySQL/PostgreSQL schema -> Qwen LoRA -> RAG -> validated SQL -> live data"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

llm = SQLGenerator()

# Warm up the RAG embedder at startup instead of lazily on the first request --
# SentenceTransformer(...) downloads/loads its weights on first use (~110s
# observed on CPU), which was previously stacking on top of the LLM's own
# slow CPU generation time inside the same request and blowing past client/
# ngrok timeouts. Loading it here means only server startup pays that cost.
_get_embedder()


@app.get("/health")
def health():
    return {"success": True, "message": "API is running"}


@app.post("/api/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(config.RATE_LIMIT)
def ask_database(request: Request, body: AskRequest):
    request_id = new_request_id()
    started = time.perf_counter()
    retrieved_tables = []
    validation_status = None
    repair_attempted = False

    try:
        engine = create_engine_from_request(body)
        schema = get_database_schema(engine)

        if not schema:
            raise HTTPException(status_code=400, detail="No tables found")

        relationships = get_table_relationships(engine)
        relevant_schema = filter_relevant_schema_rag(schema, body.question, relationships)
        retrieved_tables = list(relevant_schema.keys())

        if not relevant_schema:
            log_audit(
                request_id=request_id, status="unanswerable",
                database=body.database, retrieved_tables=retrieved_tables,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return AskResponse(
                status="unanswerable", request_id=request_id,
                message="The active schema does not contain information to answer this question.",
            )

        raw_output = llm.generate_sql(
            question=body.question,
            database_type=body.database_type or config.DB_TYPE,
            schema=relevant_schema,
        )

        sql = extract_sql(raw_output)
        sql = strip_unrequested_where(sql, body.question)
        sql = expand_to_select_star(sql, body.question, relevant_schema)

        try:
            validate_sql(sql, body.allow_limit)
            validate_sql_against_schema(sql, relevant_schema)
        except ValueError as first_error:
            repair_attempted = True
            try:
                sql = attempt_repair(
                    llm, body.question, body.database_type or config.DB_TYPE,
                    relevant_schema, sql, str(first_error), body.allow_limit,
                )
            except ValueError as repair_error:
                log_audit(
                    request_id=request_id, status="error", error_class="INVALID_SQL",
                    database=body.database, retrieved_tables=retrieved_tables,
                    repair_attempted=True, sql=sql,
                    latency_ms=round((time.perf_counter() - started) * 1000, 1),
                )
                raise HTTPException(status_code=400, detail=str(repair_error))

        sql = enforce_row_limit(sql, min(body.allow_limit, config.ALLOW_LIMIT))
        validation_status = ValidationStatus(safe=True, tables_valid=True, columns_valid=True)

        records = execute_query(engine, sql)
        formatted = normalize_result(records)
        execution_ms = round((time.perf_counter() - started) * 1000, 1)

        log_audit(
            request_id=request_id, status="success",
            database=body.database, retrieved_tables=retrieved_tables,
            sql=sql, repair_attempted=repair_attempted,
            row_count=len(records), latency_ms=execution_ms,
        )

        return AskResponse(
            status="success",
            sql=sql,
            columns=formatted["columns"],
            rows=formatted["rows"],
            row_count=len(records),
            execution_ms=execution_ms,
            validation=validation_status,
            model=config.MODEL_VERSION,
            request_id=request_id,
        )

    except HTTPException as exc:
        log_audit(
            request_id=request_id, status="error", error_class="HTTP_ERROR",
            detail=str(exc.detail), retrieved_tables=retrieved_tables,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        raise
    except Exception as exc:
        log_audit(
            request_id=request_id, status="error", error_class="INTERNAL_ERROR",
            retrieved_tables=retrieved_tables,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        raise HTTPException(status_code=500, detail="Internal server error") from exc
