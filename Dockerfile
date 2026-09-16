# Styla API — CPU-only container for a Hugging Face Docker Space, or any
# container host. Build from the repository root:
#
#   docker build -t styla-api .
#   docker run -p 7860:7860 --env-file .env styla-api
#
# The website is deployed separately (see netlify.toml and DEPLOY.md).

FROM python:3.11-slim

# libglib2.0 is needed by OpenCV; curl backs the health check below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run containers as uid 1000, and only that user's home is
# writable — so the model cache and any runtime files have to live there.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

WORKDIR $HOME/app

# Dependencies first, so code edits don't rebuild this layer. Torch comes from
# the CPU wheel index; the default wheels pull ~2 GB of unused CUDA.
COPY --chown=user requirements-serve.txt ./deps/requirements-serve.txt
COPY --chown=user backend/requirements.txt ./deps/requirements-api.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r deps/requirements-serve.txt -r deps/requirements-api.txt

# Bake the model weights into the image: a cold start then loads from disk
# instead of downloading 1.5 GB on the first request.
COPY --chown=user scripts/preload_models.py ./scripts/preload_models.py
RUN python scripts/preload_models.py

# Application code and the seeded wardrobe photos.
COPY --chown=user backend/ ./backend/
COPY --chown=user ml/ ./ml/
COPY --chown=user data/ ./data/

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-7860}/health" || exit 1

# One worker: each would load its own copy of the models.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
