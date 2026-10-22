FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 bookwriter \
    && mkdir /data && chown bookwriter:bookwriter /data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY book_writer.py ./
COPY book_studio ./book_studio
COPY static ./static

USER bookwriter
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "book_studio.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
