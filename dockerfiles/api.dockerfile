# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements.txt .
COPY pyproject.toml .
COPY src/group_99/ .

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["uvicorn", "src/group_99/api:app", "--host", "0.0.0.0", "--port", "8000"]
