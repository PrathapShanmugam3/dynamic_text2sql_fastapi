from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine

from app import config
from app.models import AskRequest


def _resolve_connection(request: AskRequest) -> dict:
    return {
        "database_type": request.database_type or config.DB_TYPE,
        "host": request.host or config.DB_HOST,
        "port": request.port or config.DB_PORT,
        "database": request.database or config.DB_NAME,
        "username": request.username or config.DB_USER,
        "password": request.password or config.DB_PASSWORD,
    }


def create_engine_from_request(request: AskRequest) -> Engine:
    conn = _resolve_connection(request)

    missing = [key for key, value in conn.items() if not value]
    if missing:
        raise ValueError(
            f"Missing database connection details: {', '.join(missing)} "
            "(provide in request or set corresponding DB_* env vars)"
        )

    if conn["database_type"] == "mysql":
        url = (
            f"mysql+pymysql://{conn['username']}:{conn['password']}"
            f"@{conn['host']}:{conn['port']}/{conn['database']}"
        )
    elif conn["database_type"] == "postgresql":
        url = (
            f"postgresql+psycopg2://{conn['username']}:{conn['password']}"
            f"@{conn['host']}:{conn['port']}/{conn['database']}"
        )
    else:
        raise ValueError("Unsupported database type")

    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={"connect_timeout": config.QUERY_TIMEOUT_SECONDS},
    )


def get_database_schema(engine: Engine) -> dict:
    inspector = inspect(engine)
    schema = {}

    for table_name in inspector.get_table_names():
        columns = []
        for column in inspector.get_columns(table_name):
            columns.append({
                "name": column["name"],
                "type": str(column["type"]),
                "nullable": column.get("nullable", True)
            })
        schema[table_name] = columns

    return schema


def get_table_relationships(engine: Engine) -> dict:
    """Foreign-key relationships per table, for the RAG knowledge unit
    (SRS Section 12.1)."""
    inspector = inspect(engine)
    relationships = {}

    for table_name in inspector.get_table_names():
        rels = []
        for fk in inspector.get_foreign_keys(table_name):
            referred_table = fk.get("referred_table")
            constrained = fk.get("constrained_columns") or []
            referred = fk.get("referred_columns") or []
            for local_col, remote_col in zip(constrained, referred):
                rels.append(f"{table_name}.{local_col} -> {referred_table}.{remote_col}")
        if rels:
            relationships[table_name] = rels

    return relationships
