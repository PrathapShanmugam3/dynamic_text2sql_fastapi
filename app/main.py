from fastapi import FastAPI, HTTPException
from app.models import AskRequest, AskResponse
from app.database import create_engine_from_request, get_database_schema
from app.llm import SQLGenerator
from app.sql_utils import extract_sql, validate_sql, execute_query

app = FastAPI(
    title="Dynamic Text-to-SQL API",
    version="1.0.0",
    description="Dynamic MySQL/PostgreSQL schema -> Qwen LoRA -> validated SQL -> live data"
)

llm = SQLGenerator()

@app.get("/health")
def health():
    return {"success": True, "message": "API is running"}

@app.post("/api/ask", response_model=AskResponse)
def ask_database(request: AskRequest):
    try:
        engine = create_engine_from_request(request)
        schema = get_database_schema(engine)

        if not schema:
            raise HTTPException(status_code=400, detail="No tables found")

        raw_output = llm.generate_sql(
            question=request.question,
            database_type=request.database_type,
            schema=schema
        )
        print("RAW MODEL OUTPUT:\n", raw_output)

        sql = extract_sql(raw_output)
        print("EXTRACTED SQL:\n", sql)

        validate_sql(sql, request.allow_limit)

        data = execute_query(engine, sql)

        return AskResponse(
            success=True,
            sql=sql,
            count=len(data),
            data=data
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
