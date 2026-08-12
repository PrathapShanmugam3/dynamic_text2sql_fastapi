from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from app.models import AskRequest

def create_engine_from_request(request: AskRequest) -> Engine:
    if request.database_type == "mysql":
        url = (
            f"mysql+pymysql://{request.username}:{request.password}"
            f"@{request.host}:{request.port}/{request.database}"
        )
    elif request.database_type == "postgresql":
        url = (
            f"postgresql+psycopg2://{request.username}:{request.password}"
            f"@{request.host}:{request.port}/{request.database}"
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
