"""Minimal traffic dashboard: volume table, alert feed, camera health.

Reads the stream processor's output artifacts (volume CSV + alerts JSONL) — no
Kafka connection needed, so it works identically for live runs and replays.

Usage:
    uv run uvicorn src.dashboard.app:app --reload
    # or: uv run python -m src.dashboard.app
"""

from __future__ import annotations

import csv
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.streaming.outputs import read_alerts

STALE_AFTER_S = 30 * 60  # camera considered quiet on the dashboard after 30 min


def read_volume_rows(csv_path: Path, limit: int = 200) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    return rows[-limit:][::-1]  # newest first


def camera_health(rows: list[dict], now: datetime | None = None) -> list[dict]:
    """Latest window seen per camera, flagged quiet when too old."""
    now = now or datetime.now(UTC)
    latest: dict[str, str] = {}
    for row in rows:  # rows are newest-first
        latest.setdefault(row["camera_id"], row["window_end"])
    health = []
    for camera_id, window_end in sorted(latest.items()):
        age_s = (now - datetime.fromisoformat(window_end)).total_seconds()
        health.append(
            {
                "camera_id": camera_id,
                "last_window_end": window_end,
                "status": "ok" if age_s < STALE_AFTER_S else "quiet",
            }
        )
    return health


def create_app(volume_csv: Path, alerts_jsonl: Path) -> FastAPI:
    app = FastAPI(title="Traffic Flow Dashboard")

    @app.get("/api/volumes")
    def api_volumes(limit: int = 200) -> list[dict]:
        return read_volume_rows(volume_csv, limit)

    @app.get("/api/alerts")
    def api_alerts(limit: int = 100) -> list[dict]:
        alerts = read_alerts(alerts_jsonl)[-limit:][::-1]
        return [a.model_dump(mode="json") for a in alerts]

    @app.get("/api/health")
    def api_health() -> list[dict]:
        return camera_health(read_volume_rows(volume_csv))

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        rows = read_volume_rows(volume_csv, limit=48)
        alerts = read_alerts(alerts_jsonl)[-20:][::-1]
        health = camera_health(read_volume_rows(volume_csv))

        def table(headers: list[str], body: list[list[str]]) -> str:
            head = "".join(f"<th>{h}</th>" for h in headers)
            rows_html = "".join(
                "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in body
            )
            return f"<table><tr>{head}</tr>{rows_html}</table>"

        health_html = table(
            ["camera", "last window", "status"],
            [[h["camera_id"], h["last_window_end"], h["status"]] for h in health],
        )
        volumes_html = table(
            ["window start", "camera", "dir", "class", "count"],
            [
                [r["window_start"], r["camera_id"], r["direction"], r["vehicle_class"], r["count"]]
                for r in rows
            ],
        )
        alerts_html = table(
            ["time", "camera", "type", "message"],
            [
                [a.ts_alert.strftime("%H:%M:%SZ"), a.camera_id, a.alert_type, a.message]
                for a in alerts
            ],
        )
        return f"""
        <html><head><title>Traffic Flow Dashboard</title><style>
          body {{ font-family: system-ui, sans-serif; margin: 2rem;
                  background: #111; color: #eee; }}
          h1 {{ font-size: 1.3rem; }} h2 {{ font-size: 1rem; margin-top: 2rem; color: #9cf; }}
          table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
          th, td {{ border: 1px solid #333; padding: 4px 8px; text-align: left; }}
          th {{ background: #1c1c1c; }}
        </style></head><body>
          <h1>Traffic Flow — 15-minute volumes &amp; alerts</h1>
          <h2>Camera health</h2>{health_html}
          <h2>Alerts (latest 20)</h2>{alerts_html or "<p>none</p>"}
          <h2>Volume windows (latest 48 rows)</h2>{volumes_html or "<p>no data yet</p>"}
        </body></html>
        """

    return app


load_dotenv()
app = create_app(
    Path(os.environ.get("VOLUME_CSV", "outputs/volumes.csv")),
    Path(os.environ.get("ALERTS_JSONL", "outputs/alerts.jsonl")),
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
