from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunInfo:
    run_id: str
    pipeline: str
    tier: str
    mode: str
    status: str
    started_at: str
    ended_at: str | None = None
    config_json: str = "{}"
    summary_json: str = "{}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: str):
    p = Path(db_path)
    # Existing projects may already have a SQLite-formatted metrics file.
    # DuckDB can open it through sqlite scanner but does not support all write ops.
    if p.exists():
        try:
            header = p.read_bytes()[:16]
            if header.startswith(b"SQLite format 3"):
                return sqlite3.connect(str(db_path)), "sqlite"
        except Exception:
            pass
    try:
        import duckdb  # type: ignore

        return duckdb.connect(str(db_path)), "duckdb"
    except Exception:
        # Fallback keeps local-first flow alive when duckdb is not installed.
        return sqlite3.connect(str(db_path)), "sqlite"


def init_db(db_path: str) -> None:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con, _backend = _connect(str(p))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            pipeline TEXT,
            tier TEXT,
            mode TEXT,
            status TEXT,
            started_at TEXT,
            ended_at TEXT,
            config_json TEXT,
            summary_json TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            run_id TEXT,
            condition TEXT,
            metric_name TEXT,
            metric_value DOUBLE,
            recorded_at TEXT
        )
        """
    )
    try:
        con.commit()
    except Exception:
        pass
    con.close()


def start_run(
    db_path: str,
    run_id: str,
    pipeline: str,
    tier: str,
    mode: str,
    config_obj: dict[str, Any],
) -> None:
    con, _backend = _connect(db_path)
    info = RunInfo(
        run_id=run_id,
        pipeline=pipeline,
        tier=tier,
        mode=mode,
        status="RUNNING",
        started_at=_utc_now_iso(),
        config_json=json.dumps(config_obj, ensure_ascii=True),
    )
    con.execute(
        """
        INSERT OR REPLACE INTO runs
        (run_id, pipeline, tier, mode, status, started_at, ended_at, config_json, summary_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            info.run_id,
            info.pipeline,
            info.tier,
            info.mode,
            info.status,
            info.started_at,
            info.ended_at,
            info.config_json,
            info.summary_json,
        ],
    )
    try:
        con.commit()
    except Exception:
        pass
    con.close()


def log_metric(
    db_path: str,
    run_id: str,
    condition: str,
    metric_name: str,
    metric_value: float,
) -> None:
    con, _backend = _connect(db_path)
    con.execute(
        """
        INSERT INTO metrics (run_id, condition, metric_name, metric_value, recorded_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [run_id, condition, metric_name, float(metric_value), _utc_now_iso()],
    )
    try:
        con.commit()
    except Exception:
        pass
    con.close()


def end_run(
    db_path: str,
    run_id: str,
    status: str,
    summary_obj: dict[str, Any],
) -> None:
    con, _backend = _connect(db_path)
    con.execute(
        """
        UPDATE runs
        SET status = ?, ended_at = ?, summary_json = ?
        WHERE run_id = ?
        """,
        [status, _utc_now_iso(), json.dumps(summary_obj, ensure_ascii=True), run_id],
    )
    try:
        con.commit()
    except Exception:
        pass
    con.close()
