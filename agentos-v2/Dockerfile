# Pinned to 3.12 — do NOT change to 3.14 or 3.13.
# grpcio (required by langchain-google-genai) has no pre-built wheels
# for Python 3.13+ pre-release versions.
FROM python:3.12-slim

WORKDIR /app

# gcc: needed for some native wheels (duckduckgo-search, aiosqlite C extension)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# SQLite DB lives in /data — mount a persistent volume here on Railway/Render.
# Set DATABASE_URL=sqlite+aiosqlite:////data/agentos.db in your environment.
RUN mkdir -p /data

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
