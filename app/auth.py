from fastapi import Header, HTTPException

from app import config


def require_api_key(x_api_key: str = Header(default=None)):
    """API key auth (SRS Section 19). Disabled when API_KEYS is unset, so
    local/dev usage without a key configured keeps working -- set API_KEYS
    before exposing this service beyond a trusted network."""
    if not config.API_KEYS:
        return
    if not x_api_key or x_api_key not in config.API_KEYS:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
