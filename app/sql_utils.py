import re
from sqlalchemy import text
from sqlalchemy.engine import Engine

FORBIDDEN = [
    "INSERT", "UPDATE", "DELETE", "DROP",
    "ALTER", "TRUNCATE", "CREATE", "GRANT",
    "REVOKE", "EXEC", "CALL"
]

def extract_sql(text_output: str) -> str:
    text_output = text_output.strip()

    if "SQL:" in text_output:
        text_output = text_output.split("SQL:", 1)[1].strip()

    match = re.search(r"```(?:sql)?\s*(.*?)```", text_output, re.I | re.S)
    if match:
        text_output = match.group(1).strip()

    # Remove accidental leading prose before SELECT/WITH.
    match = re.search(r"\b(SELECT|WITH)\b", text_output, re.I)
    if match:
        text_output = text_output[match.start():]

    return text_output.strip().rstrip(";") + ";"

def validate_sql(sql: str, max_limit: int = 1000):
    cleaned = sql.strip().rstrip(";").strip()
    upper = cleaned.upper()

    if not (upper.startswith("SELECT ") or upper.startswith("WITH ")):
        raise ValueError("Only SELECT/WITH queries are allowed")

    for keyword in FORBIDDEN:
        if re.search(rf"\b{keyword}\b", upper):
            raise ValueError(f"Forbidden SQL operation: {keyword}")

    # Prevent multiple statements.
    if ";" in cleaned:
        raise ValueError("Multiple SQL statements are not allowed")

    # Basic safety: disallow comments in generated SQL.
    if "--" in cleaned or "/*" in cleaned or "*/" in cleaned:
        raise ValueError("SQL comments are not allowed")

    return True

def execute_query(engine: Engine, sql: str):
    with engine.connect() as connection:
        result = connection.execute(text(sql))
        return [dict(row) for row in result.mappings().all()]
