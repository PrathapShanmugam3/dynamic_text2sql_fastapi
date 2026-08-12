import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from app.models import AskRequest

def _resolve_connection(request: AskRequest) -> dict:
    return {
        "database_type": request.database_type or os.getenv("DB_TYPE"),
        "host": request.host or os.getenv("DB_HOST"),
        "port": request.port or int(os.getenv("DB_PORT", "0") or 0),
        "database": request.database or os.getenv("DB_NAME"),
        "username": request.username or os.getenv("DB_USER"),
        "password": request.password or os.getenv("DB_PASSWORD"),
    }

def create_engine_from_request(request: AskRequest) -> Engine:
    conn = _resolve_connection(request)

    missing = [key for key, value in conn.items() if not value]
    if missing:
        raise ValueError(
            f"Missing database connection details: {', '.join(missing)} "
            "(provide in request or set corresponding DB_* env vars)"
        )

    pw = conn["password"] or ""
    masked_pw = (pw[:2] + "..." + pw[-2:] + f" (len={len(pw)})") if len(pw) > 4 else "***"
    print(
        f"DB CONNECT DEBUG: type={conn['database_type']} host={conn['host']} "
        f"port={conn['port']} db={conn['database']} user={conn['username']} "
        f"password={masked_pw}"
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

    return create_engine(url, pool_pre_ping=True, pool_recycle=1800)

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
