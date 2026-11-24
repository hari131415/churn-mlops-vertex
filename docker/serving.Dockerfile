FROM python:3.11-slim AS build

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels ".[serve]"

FROM python:3.11-slim

# libgomp is needed by xgboost's prebuilt wheel
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY configs ./configs

ENV PYTHONUNBUFFERED=1 \
    CHURN_CONFIG=/app/configs/config.yaml \
    PORT=8080

USER app
EXPOSE 8080

# Single worker per container; Cloud Run scales out by instances instead.
CMD exec uvicorn churn.serve.app:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-access-log
