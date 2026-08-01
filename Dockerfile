FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REFORMULATION_DB_PATH=/app/data/reformulation_assurance_v06.db \
    REFORMULATION_ARTIFACT_ROOT=/app/data/artifacts

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/data/artifacts && useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
