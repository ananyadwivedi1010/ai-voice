# Dockerfile — containerized FastAPI server for the voice agent
# Runs the API only (not the simulation, evaluation, or voice demo)

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if needed for audio libs, but API doesn't need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.json .
COPY personas.json .
COPY prompts/ prompts/
COPY src/ src/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the API with uvicorn
# Note: ANTHROPIC_API_KEY must be provided at runtime via -e flag
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
