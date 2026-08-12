# Dynamic Text-to-SQL FastAPI

This project connects dynamically to MySQL or PostgreSQL, reads the live database schema, sends the schema + question to a Qwen 2.5 3B LoRA model, validates the generated SQL, executes it as read-only SQL, and returns live JSON data.

## Important

This project expects your already-trained LoRA checkpoint:

`texttosql/results/checkpoint-3480`

It does NOT retrain the model.

## 1. Project layout

```text
dynamic_text2sql_fastapi/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── database.py
│   ├── llm.py
│   └── sql_utils.py
├── models/
│   └── texttosql/
│       └── results/
│           └── checkpoint-3480/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## 2. Copy your model

From Google Drive, copy:

`MyDrive/texttosql/results/checkpoint-3480`

to:

`models/texttosql/results/checkpoint-3480`

The folder must contain at least:

- adapter_model.safetensors
- adapter_config.json
- tokenizer.json
- tokenizer_config.json

## 3. Run directly on a GPU machine

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export ADAPTER_PATH=/absolute/path/to/texttosql/results/checkpoint-3480

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

`http://localhost:8000/docs`

## 4. API

### POST `/api/ask`

Example:

```json
{
  "database_type": "mysql",
  "host": "127.0.0.1",
  "port": 3306,
  "database": "company",
  "username": "root",
  "password": "root",
  "question": "Get all employees",
  "allow_limit": 1000
}
```

PostgreSQL example:

```json
{
  "database_type": "postgresql",
  "host": "127.0.0.1",
  "port": 5432,
  "database": "company",
  "username": "postgres",
  "password": "postgres",
  "question": "Show active employees",
  "allow_limit": 1000
}
```

## 5. Response

```json
{
  "success": true,
  "sql": "SELECT ...;",
  "count": 25,
  "data": [
    {
      "id": 1,
      "employee_name": "John"
    }
  ]
}
```

## 6. Docker + GPU

Place the model checkpoint at:

`./models/texttosql/results/checkpoint-3480`

Then:

```bash
docker compose up --build
```

The API is available at:

`http://localhost:8000/docs`

You need an NVIDIA GPU with a compatible NVIDIA Container Toolkit for the Docker GPU setup.

## 7. Security

The included validator is a basic safety layer and permits SELECT/WITH only. It is not a complete SQL security boundary.

For production:
- use a database user with SELECT-only permissions
- never use a production database administrator account
- don't expose database passwords to the frontend
- store connection credentials securely
- restrict accessible schemas/tables
- add query timeouts
- add row limits
- add audit logging
- consider a SQL parser/AST validator
- add a schema retrieval layer for large databases
