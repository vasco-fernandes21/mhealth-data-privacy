from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .api import routes
from .data.loader import data_loader

app = FastAPI(title=settings.PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST API only (WebSocket removed for robustness)
app.include_router(routes.router, prefix=settings.API_V1_STR)

@app.on_event("startup")
async def startup_event():
    # Preload data to RAM
    try:
        data_loader.load_dataset("wesad")
        data_loader.load_dataset("sleep-edf")
        print("✅ Data Loaded Successfully")
    except Exception as e:
        print(f"⚠️ Data Loading Warning: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

