from typing import Optional
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from clerk_backend_api.jwks_helpers import verify_token, VerifyTokenOptions
from .config import config

class ClerkAuthMiddleware(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[str]:
        # Check if we're in development mode
        """ if config.ENVIRONMENT == "dev":
            return "test_user_id" """

        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        if not credentials or credentials.scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        
        token = credentials.credentials
        
        try:
            claims = verify_token(
                token,
                VerifyTokenOptions(
                    authorized_parties=["http://localhost:3000"],
                    secret_key=config.CLERK_SECRET_KEY
                )
            )
            request.state.user_id = claims.get("sub")
            request.state.session = claims
            return request.state.user_id

        except Exception as e:
            raise HTTPException(
                status_code=401, 
                detail=f"Authentication failed: {str(e)}"
            )

auth_middleware = ClerkAuthMiddleware()