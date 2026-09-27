"""
SQLite persistence layer. One file, data/app.db -- no server to run.

Design principle (PS requirement): reviewer feedback NEVER overwrites the
original prediction. `predictions.label`/`confidence`/etc are written once
by save_prediction() and never touched again. Reviewer actions only:
  (a) insert a new row into `feedback` (append-only, full audit trail)
  (b) update `predictions.review_status` (a lifecycle flag: pending ->
      confirmed/dismissed/relabeled -- NOT the ground-truth label itself)
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS batches (
            id TEXT PRIMARY KEY,
            filename TEXT,
            submitted_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            title TEXT,
            source TEXT,
            author TEXT,
            label TEXT NOT NULL,              -- model output, IMMUTABLE after creation
            confidence REAL NOT NULL,
            review_priority TEXT,
            prob_real REAL,
            prob_fake REAL,
            source_credibility REAL,
            source_known INTEGER,
            author_known INTEGER,
            main_signal TEXT,
            main_signal_direction TEXT,
            explanation_json TEXT,            -- JSON-encoded list of {feature, contribution}
            review_status TEXT NOT NULL DEFAULT 'pending',  -- pending|confirmed|dismissed|relabeled -- MUTABLE lifecycle flag only
            batch_id TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (batch_id) REFERENCES batches(id)
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            prediction_id TEXT NOT NULL,
            action TEXT NOT NULL,             -- confirm|dismiss|relabel
            corrected_label TEXT,             -- only set when action == relabel
            note TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        );
    """)
    conn.commit()
    conn.close()


def _now():
    return datetime.now(timezone.utc).isoformat()


def create_batch(filename: str) -> str:
    batch_id = str(uuid.uuid4())
    conn = _connect()
    conn.execute("INSERT INTO batches (id, filename, submitted_at) VALUES (?, ?, ?)",
                 (batch_id, filename, _now()))
    conn.commit()
    conn.close()
    return batch_id


def save_prediction(text, title, source, author, result, explanation, batch_id=None) -> str:
    """result = the dict from model/inference.py's predict(); explanation = list of dicts."""
    pred_id = str(uuid.uuid4())
    conn = _connect()
    conn.execute("""
        INSERT INTO predictions (
            id, text, title, source, author, label, confidence, review_priority,
            prob_real, prob_fake, source_credibility, source_known, author_known,
            main_signal, main_signal_direction, explanation_json, review_status,
            batch_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
    """, (
        pred_id, text, title, source, author,
        result["label"], result["confidence"], result["review_priority"],
        result["prob_real"], result["prob_fake"],
        round(1 - result["source_fake_ratio"], 4), int(result["source_known"]), int(result["author_known"]),
        (result["shap"] or {}).get("linguistic_signal") is not None and _main_signal(result) or None,
        _main_signal_direction(result),
        json.dumps(explanation),
        batch_id, _now(),
    ))
    conn.commit()
    conn.close()
    return pred_id


def _main_signal(result):
    shap = result.get("shap")
    if not shap:
        return None
    strengths = {
        "linguistic": shap["linguistic_strength"],
        "semantic": shap["semantic_strength"],
        "credibility": shap["credibility_strength"],
    }
    return max(strengths, key=strengths.get)


def _main_signal_direction(result):
    shap = result.get("shap")
    if not shap:
        return None
    signal = _main_signal(result)
    value = shap[f"{signal}_signal"]
    return "real" if value > 0 else ("fake" if value < 0 else "neutral")


def get_predictions(review_status=None, label=None, source=None, batch_id=None, no_batch=False,
                     sort_by="confidence", sort_dir="asc", limit=200):
    """For the reviewer queue. Sorting by confidence ascending surfaces the
    lowest-confidence (most-in-need-of-review) items first by default.
    no_batch=True overrides batch_id and returns only ungrouped single-analysis items."""
    conn = _connect()
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []
    if review_status:
        query += " AND review_status = ?"
        params.append(review_status)
    if label:
        query += " AND label = ?"
        params.append(label)
    if source:
        query += " AND source LIKE ?"
        params.append(f"%{source}%")
    if no_batch:
        query += " AND batch_id IS NULL"
    elif batch_id:
        query += " AND batch_id = ?"
        params.append(batch_id)

    allowed_sort_cols = {"confidence", "created_at", "label", "source"}
    if sort_by not in allowed_sort_cols:
        sort_by = "confidence"
    sort_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"
    query += f" ORDER BY {sort_by} {sort_dir} LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_prediction(prediction_id: str):
    conn = _connect()
    row = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_feedback_history(prediction_id: str):
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM feedback WHERE prediction_id = ? ORDER BY created_at ASC",
        (prediction_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_feedback(prediction_id: str, action: str, corrected_label: str = None, note: str = None):
    """action: 'confirm' | 'dismiss' | 'relabel'. Inserts an audit row, then updates
    ONLY the review_status lifecycle flag on the prediction -- never label/confidence."""
    if action not in ("confirm", "dismiss", "relabel"):
        raise ValueError(f"Invalid action: {action}")
    if action == "relabel" and not corrected_label:
        raise ValueError("corrected_label is required when action='relabel'")

    conn = _connect()
    feedback_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO feedback (id, prediction_id, action, corrected_label, note, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (feedback_id, prediction_id, action, corrected_label, note, _now()),
    )

    new_status = {"confirm": "confirmed", "dismiss": "dismissed", "relabel": "relabeled"}[action]
    conn.execute("UPDATE predictions SET review_status = ? WHERE id = ?", (new_status, prediction_id))
    conn.commit()
    conn.close()
    return feedback_id


def _batch_stats(conn, batch_id):
    if batch_id is None:
        rows = conn.execute("SELECT label, review_status FROM predictions WHERE batch_id IS NULL").fetchall()
    else:
        rows = conn.execute("SELECT label, review_status FROM predictions WHERE batch_id = ?", (batch_id,)).fetchall()
    label_counts, status_counts = {}, {}
    for r in rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
        status_counts[r["review_status"]] = status_counts.get(r["review_status"], 0) + 1
    return {"total_items": len(rows), "label_counts": label_counts, "status_counts": status_counts}


def get_batches():
    """Returns every batch upload plus a pseudo-group for ungrouped single-analysis
    items (batch_id IS NULL), each with item/label/review-status counts. Used to
    render the Review Queue's group list before drilling into one."""
    conn = _connect()
    batch_rows = conn.execute("SELECT id, filename, submitted_at FROM batches ORDER BY submitted_at DESC").fetchall()
    batches = []
    for b in batch_rows:
        stats = _batch_stats(conn, b["id"])
        if stats["total_items"] == 0:
            continue
        batches.append({"id": b["id"], "filename": b["filename"], "submitted_at": b["submitted_at"], **stats})

    single_stats = _batch_stats(conn, None)
    if single_stats["total_items"] > 0:
        batches.append({"id": None, "filename": "(Single Analysis — ungrouped)", "submitted_at": None, **single_stats})

    conn.close()
    return batches


def get_batch_summary(batch_id: str):
    conn = _connect()
    rows = conn.execute("SELECT label FROM predictions WHERE batch_id = ?", (batch_id,)).fetchall()
    conn.close()
    total = len(rows)
    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    return {"batch_id": batch_id, "total_items": total, "label_counts": counts}
