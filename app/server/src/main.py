from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .core.database import init_db
from .api import routes
from .data.loader import data_loader


app = FastAPI(title=settings.PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    # Em dev aceitamos qualquer origem para simplificar, mas SEM credenciais
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
        print("✅ Data Loaded Successfully")
    except Exception as e:
        print(f"⚠️ Data Loading Warning: {e}")


@app.get("/health")
def health_check():
    return {"status": "ok"}

