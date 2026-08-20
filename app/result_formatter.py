"""Normalizes raw DB rows into the columns/rows shape from SRS Section 15.2."""
import datetime
import decimal


def _normalize_value(value):
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def normalize_result(records: list) -> dict:
    if not records:
        return {"columns": [], "rows": []}

    columns = list(records[0].keys())
    rows = [[_normalize_value(record[col]) for col in columns] for record in records]
    return {"columns": columns, "rows": rows}
