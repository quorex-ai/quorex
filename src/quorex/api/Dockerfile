FROM python:3.12-slim
WORKDIR /app

# Install deps first (better layer caching), then the package.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir numpy fastapi "uvicorn[standard]" \
 && pip install --no-cache-dir -e .

# Engine persists WAL/snapshots + encoder here (mounted volume in compose).
ENV QUOREX_DB_PATH=/data/quorex_memory \
    QUOREX_ENCODER_PATH=/data/quorex_encoder
EXPOSE 8000

CMD ["python", "src/quorex/api/run.py"]