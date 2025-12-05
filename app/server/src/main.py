from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

