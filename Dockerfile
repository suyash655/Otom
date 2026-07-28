# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS backend

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and checkpoints
COPY src/ ./src/
COPY backend/ ./backend/
COPY checkpoints/ ./checkpoints/
COPY data/splits.json ./data/splits.json

# Expose FastAPI port
EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
