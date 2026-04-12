# ── Stage 1: build dependencies ────────────────────────────────────────────
FROM python:3.11-slim AS builder
 
WORKDIR /build
 
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt
 
# ── Stage 2: runtime image ──────────────────────────────────────────────────
FROM python:3.11-slim
 
# Non-root user for security
RUN useradd -m -u 1000 appuser
 
WORKDIR /app
 
# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
 
# Copy application source
COPY . .
 
# Ensure correct ownership
RUN chown -R appuser:appuser /app
USER appuser
 
# HF Spaces exposes port 7860
EXPOSE 7860
 
# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1
 
# Default: run the FastAPI server
CMD ["python", "-m", "uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "7860"]