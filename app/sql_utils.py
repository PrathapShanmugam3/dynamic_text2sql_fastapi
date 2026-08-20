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

    text_output = text_output.strip().rstrip(";").strip()

    # Cut at the first statement terminator or paragraph break -- the model
    # sometimes rambles into unrelated prose (or fake "REFUSED:"/GRANT text)
    # after a complete statement, with no LIMIT clause to anchor on.
    match = re.search(r";|\n\s*\n", text_output)
    if match:
        text_output = text_output[:match.start()]

    # Truncate hallucinated trailing tokens after a valid LIMIT/OFFSET clause
    # (e.g. "LIMIT 50 OFFSET 0 ROWS FETCH NEXT 49 AHEAD" -> "LIMIT 50 OFFSET 0").
    match = re.search(
        r"\bLIMIT\s+\d+(?:\s+OFFSET\s+\d+)?", text_output, re.I
    )
    if match:
        text_output = text_output[:match.end()]

    return text_output.strip().rstrip(";") + ";"

FIELD_HINT_WORDS = [
    "name", "email", "mobile", "phone", "id", "count", "total",
    "sum", "average", "avg", "max", "min", "only", "column",
]

def expand_to_select_star(sql: str, question: str) -> str:
    """The fine-tuned model tends to emit SELECT id ... for open-ended
    'get all X' questions even when told to select all columns. Detect
    that narrow case and widen it to SELECT * when the question names
    no specific fields and the query has no WHERE/GROUP BY/aggregates."""
    question_lower = question.lower()
    if any(hint in question_lower for hint in FIELD_HINT_WORDS):
        return sql

    match = re.match(r"SELECT\s+`?id`?\s+FROM\s+(`?\w+`?)(.*)", sql, re.I | re.S)
    if not match:
        return sql

    tail = match.group(2)
    if re.search(r"\b(WHERE|GROUP BY|HAVING|JOIN)\b", tail, re.I):
        return sql

    return f"SELECT * FROM {match.group(1)}{tail}"

FILTER_HINT_WORDS = [
    "where", "whose", "that has", "that have",
    "greater", "less", "more than", "at least", "at most",
    "before", "after", "between", "since", "until",
    "equal", "not equal", "contains", "like",
    "is active", "is inactive", "with status", "of type",
]

VALUE_PATTERN = re.compile(
    r"""['"][^'"]+['"]"""          # quoted string, e.g. 'pynixindia@gmail.com'
    r"""|\b[\w.+-]+@[\w-]+\.[\w.-]+\b"""  # bare email address
    r"""|\b\d+\b"""                # standalone number/id
)

def strip_unrequested_where(sql: str, question: str) -> str:
    """Drop a WHERE clause the model added when the question gave no filter criteria."""
    question_lower = question.lower()
    if any(hint in question_lower for hint in FILTER_HINT_WORDS):
        return sql
    if VALUE_PATTERN.search(question):
        return sql

    match = re.search(r"\bWHERE\b", sql, re.I)
    if not match:
        return sql

    tail_match = re.search(
        r"\b(GROUP BY|ORDER BY|LIMIT|HAVING)\b", sql[match.end():], re.I
    )
    if tail_match:
        tail_start = match.end() + tail_match.start()
        return (sql[:match.start()] + sql[tail_start:]).strip()

    end_match = re.search(r";\s*$", sql)
    tail = sql[end_match.start():] if end_match else ""
    return (sql[:match.start()].rstrip() + tail).strip()

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

    # Reject malformed pagination the model sometimes hallucinates
    # (mixing LIMIT/OFFSET with FETCH ... ROWS ONLY is invalid in MySQL).
    if "LIMIT" in upper and "FETCH" in upper and "ROWS ONLY" in upper:
        raise ValueError("Malformed SQL: mixed LIMIT and FETCH FIRST clauses")

    return True

def validate_sql_against_schema(sql: str, schema: dict):
    """Reject SQL that references tables/columns not present in the schema."""
    cleaned = sql.strip().rstrip(";").strip()

    known_tables = {name.lower() for name in schema}
    known_columns = set()
    for columns in schema.values():
        for col in columns:
            col_name = col["name"] if isinstance(col, dict) else col
            known_columns.add(col_name.lower())

    used_tables = {
        m.lower() for m in re.findall(
            r"\bFROM\s+`?(\w+)`?|\bJOIN\s+`?(\w+)`?", cleaned, re.I
        ) for m in m if m
    }
    unknown_tables = used_tables - known_tables
    if unknown_tables:
        raise ValueError(f"Unknown table(s) referenced: {', '.join(sorted(unknown_tables))}")

    used_columns = {c.lower() for c in re.findall(r"`(\w+)`", cleaned)} - used_tables
    unknown_columns = used_columns - known_columns
    if unknown_columns:
        raise ValueError(f"Unknown column(s) referenced: {', '.join(sorted(unknown_columns))}")

    return True

def enforce_row_limit(sql: str, max_limit: int) -> str:
    """Ensure the query has a server-side LIMIT no greater than max_limit
    (SRS FR-011 / Section 14: prefer server-side LIMIT enforcement)."""
    cleaned = sql.strip().rstrip(";").strip()
    match = re.search(r"\bLIMIT\s+(\d+)", cleaned, re.I)

    if match:
        existing = int(match.group(1))
        if existing > max_limit:
            cleaned = cleaned[:match.start(1)] + str(max_limit) + cleaned[match.end(1):]
    else:
        cleaned = f"{cleaned} LIMIT {max_limit}"

    return cleaned + ";"


def execute_query(engine: Engine, sql: str):
    with engine.connect() as connection:
        result = connection.execute(text(sql))
        return [dict(row) for row in result.mappings().all()]
