# The agent only. Ollama is expected to run on the host.
#
# Running Ollama inside a container on macOS gives up Metal and falls back to
# CPU, which takes generation from ~13 tok/s to unusable. Keeping the model
# server on the host is the difference between a demo that works and one that
# technically starts.
FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so editing source does not invalidate the install layer.
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-cache-dir -e '.[api]'

# Reach the host's Ollama from inside the container. Docker Desktop provides
# host.docker.internal; on Linux, run with --add-host or use the compose file.
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    VECTOR_STORE_PATH=/data/qdrant \
    PYTHONUNBUFFERED=1

# The index is state, not part of the image.
VOLUME ["/data"]

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
