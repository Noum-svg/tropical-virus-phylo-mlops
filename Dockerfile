# Tropical Virus PhyloTree MLOps — application image (API + React UI + CLI).

# --- Stage 1: build the React frontend ---
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Python application ---
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Copy the project, then the built frontend from stage 1.
COPY . .
COPY --from=frontend /frontend/dist ./frontend/dist

EXPOSE 8000 8501

# Default: serve the professional React UI + API at http://localhost:8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
