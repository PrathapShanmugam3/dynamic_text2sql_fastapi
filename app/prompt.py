"""Canonical Text-to-SQL prompt builder.

This is the ONLY prompt format supported for training, evaluation, and
inference (SRS FR-005 / AC-03). It must stay byte-identical to the
schema-injected `text` field baked into train.csv/validation.csv/test.csv,
since that is what the model was fine-tuned on.
"""

RULES = """Rules:
1. Generate only valid {db_type} SQL.
2. Use only tables from the provided schema.
3. Use only columns from the provided schema.
4. Never invent table names.
5. Never invent column names.
6. Generate SELECT queries only.
7. Do not generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE."""


def schema_to_text(schema: dict) -> str:
    parts = []
    for table, columns in schema.items():
        parts.append(f"TABLE {table}")
        for col in columns:
            name = col["name"] if isinstance(col, dict) else col
            col_type = col.get("type", "") if isinstance(col, dict) else ""
            parts.append(f"  - {name} {col_type}".rstrip())
        parts.append("")
    return "\n".join(parts).rstrip("\n")


def build_prompt(question: str, database_type: str, schema: dict) -> str:
    schema_text = schema_to_text(schema)
    rules = RULES.format(db_type=database_type)

    return (
        "You are a professional Text-to-SQL system.\n\n"
        f"Database type:\n{database_type}\n\n"
        f"Database schema:\n{schema_text}\n\n"
        f"{rules}\n\n"
        f"User question:\n{question}\n\n"
        "SQL:\n"
    )


def build_repair_prompt(question: str, database_type: str, schema: dict, bad_sql: str, error: str) -> str:
    """One controlled repair attempt (SRS Section 13.2). Reuses the canonical
    prompt shape and appends the failed SQL plus the validation/DB error."""
    base = build_prompt(question, database_type, schema)
    return (
        base.rstrip() + "\n\n"
        f"The previous SQL attempt was invalid:\n{bad_sql}\n\n"
        f"Validation error:\n{error}\n\n"
        "Fix the SQL using only the tables and columns above. Return corrected SQL only.\n\nSQL:\n"
    )
