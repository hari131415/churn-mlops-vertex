"""Ship request features + predictions to BigQuery.

Vertex Model Monitoring reads this table as its target dataset, which is how we
get drift/skew monitoring for a model that isn't on a Vertex endpoint.

Writes happen off the request path in a background thread and are batched, so
a slow BigQuery insert never shows up in p95.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import UTC, datetime
from typing import Protocol

log = logging.getLogger(__name__)


class PredictionSink(Protocol):
    def write(self, rows: list[dict]) -> None: ...
    def close(self) -> None: ...


class NullSink:
    def write(self, rows: list[dict]) -> None:
        pass

    def close(self) -> None:
        pass


class BigQuerySink:
    def __init__(self, table_id: str, flush_every: int = 200, flush_seconds: float = 5.0):
        from google.cloud import bigquery

        self._client = bigquery.Client()
        self._table_id = table_id
        self._q: queue.Queue[dict] = queue.Queue(maxsize=50_000)
        self._flush_every = flush_every
        self._flush_seconds = flush_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bq-sink", daemon=True)
        self._thread.start()

    def write(self, rows: list[dict]) -> None:
        for row in rows:
            try:
                self._q.put_nowait(row)
            except queue.Full:
                # Dropping monitoring rows is better than blocking predictions.
                log.warning("prediction log queue full, dropping row")
                return

    def _drain(self) -> list[dict]:
        batch = []
        while len(batch) < self._flush_every:
            try:
                batch.append(self._q.get_nowait())
            except queue.Empty:
                break
        return batch

    def _run(self) -> None:
        last = time.monotonic()
        while not self._stop.is_set():
            if (
                self._q.qsize() >= self._flush_every
                or time.monotonic() - last > self._flush_seconds
            ):
                self._flush()
                last = time.monotonic()
            time.sleep(0.2)
        self._flush()

    def _flush(self) -> None:
        batch = self._drain()
        if not batch:
            return
        errors = self._client.insert_rows_json(self._table_id, batch)
        if errors:
            log.error("bigquery insert errors: %s", errors[:3])

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)


def to_rows(instances: list[dict], probs: list[float], model_version: str) -> list[dict]:
    ts = datetime.now(UTC).isoformat()
    return [
        {**inst, "churn_probability": p, "model_version": model_version, "prediction_ts": ts}
        for inst, p in zip(instances, probs, strict=True)
    ]
