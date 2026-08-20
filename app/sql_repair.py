"""Single controlled SQL repair attempt (SRS Section 13.2 / FR-014).

Only one repair attempt is allowed. Multiple uncontrolled retries increase
latency and make failures harder to diagnose (SRS 13.2).
"""
from app import prompt as prompt_builder
from app.sql_utils import extract_sql, validate_sql, validate_sql_against_schema


def attempt_repair(llm, question: str, database_type: str, schema: dict, bad_sql: str, error: str, allow_limit: int):
    """Ask the model once to fix `bad_sql` given `error`. Returns the
    repaired, validated SQL string, or raises ValueError if still invalid."""
    repair_prompt = prompt_builder.build_repair_prompt(question, database_type, schema, bad_sql, error)
    raw_output = llm.generate_from_prompt(repair_prompt)
    repaired_sql = extract_sql(raw_output)

    validate_sql(repaired_sql, allow_limit)
    validate_sql_against_schema(repaired_sql, schema)

    return repaired_sql
