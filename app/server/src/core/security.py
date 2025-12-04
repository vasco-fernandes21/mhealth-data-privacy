from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

API_KEY_HEADER = APIKeyHeader(name="X-Session-ID", auto_error=False)


async def validate_session(session_id: str = Security(API_KEY_HEADER)):
    """Simple session validation. In prod, check against Redis/DB."""
    if not session_id:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
        )
    return session_id

