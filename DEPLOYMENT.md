# Deployment Guide

## Prerequisites

- Python 3.12+
- Node.js 18+
- Preprocessed datasets in `data/processed/`

## Backend Setup

### Development

```bash
cd app/server
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env with your configuration

# Run migrations
python -c "from src.core.database import init_db; init_db()"

# Start server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# Use production WSGI server
pip install gunicorn
gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Frontend Setup

### Development

```bash
cd app/client
npm install
npm run dev
```

### Production Build

```bash
npm run build
# Serve dist/ directory with nginx or similar
```

## Environment Variables

See `app/server/.env.example` for all available configuration options.

## Database

The application uses SQLite by default. For production, consider PostgreSQL:

```python
# In src/core/database.py
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/dbname")
engine = create_engine(DATABASE_URL)
```

## API Documentation

Once the server is running, access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Health Check

```bash
curl http://localhost:8000/health
```

## Testing

```bash
cd app/server
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

