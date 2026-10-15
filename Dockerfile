FROM python:3.14 AS build

WORKDIR /app

ENV OLLAMA_BASE_URL=http://ollama:11434/v1/

COPY main.py .
COPY flow.py .
COPY nodes.py .
COPY utils.py .
COPY api.py .
COPY static/ ./static/
COPY requirements.txt .

RUN useradd --create-home appuser

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

CMD ["python3", "api.py"]

FROM chromadb/chroma:latest AS croma

RUN apt update && apt install curl -y