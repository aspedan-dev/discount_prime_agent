FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DPA_DATA_PATH=/app/data/sample-data-mongo.json \
    DPA_OUT_DIR=/app/outputs

COPY requirements.txt pyproject.toml ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install .

COPY data ./data

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/outputs \
    && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["python"]
CMD ["-m", "discount_prime_agent.main", "--mode", "pipeline"]
