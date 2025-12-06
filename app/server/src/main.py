from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from .core.config import settings
from .core.database import init_db
from .api import routes
from .data.loader import data_loader


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Privacy-preserving ML training API for mHealth data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix=settings.API_V1_STR)

base_dir = Path(__file__).resolve().parent.parent.parent.parent
docs_dir = base_dir / "app" / "api_docs" / "docs"
openapi_file = base_dir / "app" / "api_docs" / "openapi.yaml"

if docs_dir.exists():
    @app.get("/api-docs", response_class=FileResponse)
    async def api_docs():
        return FileResponse(docs_dir / "index.html")
    
    @app.get("/api-docs/openapi.yaml", response_class=FileResponse)
    async def openapi_spec():
        if openapi_file.exists():
            return FileResponse(openapi_file)
        raise HTTPException(status_code=404, detail="OpenAPI spec not found")


@app.on_event("startup")
async def startup_event():
    init_db()
    try:
        data_loader.load_dataset("wesad")
        data_loader.load_dataset("sleep-edf")
    except Exception:
        pass


@app.get("/health")
def health_check():
    return {"status": "ok"}

