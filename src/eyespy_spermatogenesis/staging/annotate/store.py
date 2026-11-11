"""SQLite label store for the staging/segmentation annotation page.

One row per (tubule, mode, annotator, origin) holds that person's current
label; every change is also appended to ``events``, so nothing is lost when
a label is changed. Writes commit immediately (autosave). ``backup`` copies
the database and a CSV snapshot into ``backups/``.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

# Decision 020: eleven classes (II and III always together) plus one non-stage.
STAGES = [
    ("I", "I", "1"),
    ("II-III", "II–III", "2"),
    ("IV", "IV", "4"),
    ("V", "V", "5"),
    ("VI", "VI", "6"),
    ("VII", "VII", "7"),
    ("VIII", "VIII", "8"),
    ("IX", "IX", "9"),
    ("X", "X", "0"),
    ("XI", "XI", "-"),
    ("XII", "XII", "="),
    ("cant_stage", "Can't stage / bad crop", "b"),
]
# Same verdicts as segmentation/review_sheet.py, so CSVs stay comparable.
VERDICTS = [
    ("good", "Good", "1"),
    ("merged", "Merged", "2"),
    ("split", "Split", "3"),
    ("partial", "Partial / edge", "4"),
    ("not_tubule", "Not a tubule", "5"),
    ("unsure", "Unsure", "6"),
    ("spill", "Outline spills", "7"),
]
MODES = {
    "staging": {v for v, *_ in STAGES},
    "segmentation": {v for v, *_ in VERDICTS},
    # The reviewer's own stage for a tubule whose model predictions are shown
    # (staging round 1+); kept apart from staging labels.
    "model_review": {v for v, *_ in STAGES},
}
PAPER = "paper"
WEB = "web"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tubules (
  tubule_id TEXT PRIMARY KEY,         -- manifest instance_id
  seq INTEGER NOT NULL,               -- display order
  source_folder TEXT NOT NULL,
  source_name TEXT NOT NULL,
  source_sha256 TEXT,
  label INTEGER NOT NULL,
  recommended INTEGER NOT NULL,
  flags TEXT NOT NULL,
  source_bounds TEXT NOT NULL,
  image TEXT NOT NULL,
  outline TEXT NOT NULL,
  display_scale REAL NOT NULL         -- displayed px per native px
);
CREATE TABLE IF NOT EXISTS labels (
  tubule_id TEXT NOT NULL REFERENCES tubules(tubule_id),
  mode TEXT NOT NULL,
  annotator TEXT NOT NULL,
  origin TEXT NOT NULL,               -- 'web' or 'paper'
  value TEXT,                         -- NULL only for an unread paper mark
  detail TEXT NOT NULL DEFAULT '{}',  -- JSON: paper read flags, snippet, checks
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (tubule_id, mode, annotator, origin)
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  at TEXT NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  tubule_id TEXT,
  mode TEXT,
  annotator TEXT,
  origin TEXT,
  value TEXT,
  previous TEXT,
  detail TEXT
);
CREATE TABLE IF NOT EXISTS model_predictions (
  run TEXT NOT NULL,                  -- review run, e.g. staging-round1-2026-11-09-01
  tubule_id TEXT NOT NULL REFERENCES tubules(tubule_id),
  ord INTEGER NOT NULL,               -- review order (shuffled)
  slot TEXT NOT NULL,                 -- 'A' or 'B' (shuffled per tubule)
  model TEXT NOT NULL,
  value TEXT NOT NULL,
  PRIMARY KEY (run, tubule_id, slot)
);
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  annotator TEXT NOT NULL,
  mode TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  export TEXT
);
"""

REVIEW_COLUMNS = [
    "review_run",
    "paper_stage",
    "model_a",
    "model_a_stage",
    "model_b",
    "model_b_stage",
]

CSV_COLUMNS = [
    "mode",
    "source_name",
    "source_sha256",
    "candidate_label",
    "tubule_id",
    "recommended",
    "annotator",
    "origin",
    "value",
    "paper_read_flags",
    "paper_transcription_checked_by",
    "created_at",
    "updated_at",
    "source_bounds",
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def clean_name(name: str) -> str:
    name = " ".join(str(name).split())
    if not name or len(name) > 60:
        raise ValueError("Name must be 1-60 characters")
    return name


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self):
        with self.lock:
            self.db.close()

    # ---- tubules -------------------------------------------------------
    def add_tubules(self, rows: list[dict]):
        with self.lock, self.db:
            self.db.executemany(
                """INSERT OR REPLACE INTO tubules VALUES
                (:tubule_id,:seq,:source_folder,:source_name,:source_sha256,:label,
                 :recommended,:flags,:source_bounds,:image,:outline,:display_scale)""",
                rows,
            )

    def tubules(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM tubules ORDER BY seq").fetchall()
        return [dict(r) for r in rows]

    # ---- labels --------------------------------------------------------
    def set_label(
        self,
        tubule_id: str,
        mode: str,
        annotator: str,
        value: str | None,
        *,
        origin: str = WEB,
        actor: str | None = None,
        detail: dict | None = None,
        action: str = "set",
    ) -> dict:
        annotator = clean_name(annotator)
        actor = clean_name(actor or annotator)
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}")
        if value is not None and value not in MODES[mode]:
            raise ValueError(f"{value!r} is not a valid {mode} label")
        if origin not in (WEB, PAPER):
            raise ValueError("origin must be web or paper")
        t = now()
        with self.lock, self.db:
            if not self.db.execute(
                "SELECT 1 FROM tubules WHERE tubule_id=?", (tubule_id,)
            ).fetchone():
                raise KeyError(tubule_id)
            old = self.db.execute(
                "SELECT value, detail FROM labels WHERE tubule_id=? AND mode=? "
                "AND annotator=? AND origin=?",
                (tubule_id, mode, annotator, origin),
            ).fetchone()
            merged = json.loads(old["detail"]) if old else {}
            merged.update(detail or {})
            self.db.execute(
                """INSERT INTO labels VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(tubule_id,mode,annotator,origin) DO UPDATE SET
                value=excluded.value, detail=excluded.detail,
                updated_at=excluded.updated_at""",
                (tubule_id, mode, annotator, origin, value, json.dumps(merged), t, t),
            )
            self.db.execute(
                "INSERT INTO events (at,actor,action,tubule_id,mode,annotator,origin,"
                "value,previous,detail) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    t,
                    actor,
                    action,
                    tubule_id,
                    mode,
                    annotator,
                    origin,
                    value,
                    old["value"] if old else None,
                    json.dumps(detail or {}),
                ),
            )
        return {"saved_at": t}

    def clear_label(self, tubule_id: str, mode: str, annotator: str) -> dict:
        annotator = clean_name(annotator)
        t = now()
        with self.lock, self.db:
            old = self.db.execute(
                "SELECT value FROM labels WHERE tubule_id=? AND mode=? AND annotator=? "
                "AND origin=?",
                (tubule_id, mode, annotator, WEB),
            ).fetchone()
            self.db.execute(
                "DELETE FROM labels WHERE tubule_id=? AND mode=? AND annotator=? "
                "AND origin=?",
                (tubule_id, mode, annotator, WEB),
            )
            self.db.execute(
                "INSERT INTO events (at,actor,action,tubule_id,mode,annotator,origin,"
                "value,previous) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    t,
                    annotator,
                    "clear",
                    tubule_id,
                    mode,
                    annotator,
                    WEB,
                    None,
                    old["value"] if old else None,
                ),
            )
        return {"saved_at": t}

    def labels(self, mode: str | None = None) -> list[dict]:
        sql = "SELECT * FROM labels"
        args: tuple = ()
        if mode:
            sql += " WHERE mode=?"
            args = (mode,)
        with self.lock:
            rows = self.db.execute(sql + " ORDER BY updated_at", args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["detail"] = json.loads(d["detail"])
            out.append(d)
        return out

    def annotators(self) -> list[str]:
        with self.lock:
            rows = self.db.execute(
                "SELECT annotator, MAX(updated_at) AS last FROM labels "
                "WHERE origin='web' GROUP BY annotator "
                "UNION SELECT annotator, MAX(started_at) FROM sessions "
                "GROUP BY annotator ORDER BY last DESC"
            ).fetchall()
        seen, names = set(), []
        for r in rows:
            if r["annotator"] not in seen:
                seen.add(r["annotator"])
                names.append(r["annotator"])
        return names

    # ---- model predictions (model review mode) ------------------------
    def add_predictions(self, run: str, rows: list[dict]):
        """Replace the predictions of one review run. Labels are untouched."""
        with self.lock, self.db:
            self.db.execute("DELETE FROM model_predictions WHERE run=?", (run,))
            self.db.executemany(
                "INSERT INTO model_predictions VALUES "
                "(:run,:tubule_id,:ord,:slot,:model,:value)",
                [{**r, "run": run} for r in rows],
            )
            self.db.execute(
                "INSERT INTO events (at,actor,action,detail) VALUES (?,?,?,?)",
                (
                    now(),
                    "owner",
                    "add_predictions",
                    json.dumps({"run": run, "rows": len(rows)}),
                ),
            )

    def predictions(self, run: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM model_predictions", ()
        if run:
            sql, args = sql + " WHERE run=?", (run,)
        with self.lock:
            rows = self.db.execute(sql + " ORDER BY run, ord, slot", args).fetchall()
        return [dict(r) for r in rows]

    def review_runs(self) -> list[str]:
        with self.lock:
            rows = self.db.execute(
                "SELECT run, MAX(rowid) AS last FROM model_predictions "
                "GROUP BY run ORDER BY last DESC"
            ).fetchall()
        return [r["run"] for r in rows]

    # ---- sessions ------------------------------------------------------
    def start_session(self, annotator: str, mode: str) -> int:
        annotator = clean_name(annotator)
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}")
        with self.lock, self.db:
            cur = self.db.execute(
                "INSERT INTO sessions (annotator, mode, started_at) VALUES (?,?,?)",
                (annotator, mode, now()),
            )
            self.db.execute(
                "INSERT INTO events (at,actor,action,mode) VALUES (?,?,?,?)",
                (now(), annotator, "login", mode),
            )
        return int(cur.lastrowid)

    def end_session(self, session_id: int, export: str | None) -> dict | None:
        with self.lock, self.db:
            row = self.db.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            if not row:
                return None
            self.db.execute(
                "UPDATE sessions SET ended_at=?, export=? WHERE id=?",
                (now(), export, session_id),
            )
            self.db.execute(
                "INSERT INTO events (at,actor,action,mode,detail) VALUES (?,?,?,?,?)",
                (
                    now(),
                    row["annotator"],
                    "logout",
                    row["mode"],
                    json.dumps({"export": export}),
                ),
            )
            return dict(row)

    def session_count(self, session_id: int) -> int:
        with self.lock:
            row = self.db.execute(
                "SELECT s.annotator, s.started_at FROM sessions s WHERE id=?",
                (session_id,),
            ).fetchone()
            if not row:
                return 0
            return self.db.execute(
                "SELECT COUNT(*) FROM events WHERE actor=? AND at>=? "
                "AND action IN ('set','transcription')",
                (row["annotator"], row["started_at"]),
            ).fetchone()[0]

    # ---- exports and backups ------------------------------------------
    def csv_text(self) -> str:
        tub = {t["tubule_id"]: t for t in self.tubules()}
        paper = {
            lab["tubule_id"]: lab["value"]
            for lab in self.labels("staging")
            if lab["origin"] == PAPER
        }
        preds: dict[tuple, dict] = {}
        for p in self.predictions():
            preds.setdefault((p["run"], p["tubule_id"]), {})[p["slot"]] = p
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(CSV_COLUMNS + REVIEW_COLUMNS)
        for lab in sorted(
            self.labels(),
            key=lambda r: (r["mode"], tub[r["tubule_id"]]["seq"], r["annotator"]),
        ):
            t = tub[lab["tubule_id"]]
            d = lab["detail"]
            w.writerow(
                [
                    lab["mode"],
                    t["source_name"],
                    t["source_sha256"],
                    t["label"],
                    lab["tubule_id"],
                    t["recommended"],
                    lab["annotator"],
                    lab["origin"],
                    lab["value"] or "",
                    ";".join(d.get("read_flags", [])),
                    d.get("transcription_checked_by", ""),
                    lab["created_at"],
                    lab["updated_at"],
                    t["source_bounds"],
                    *self._review_cells(lab, paper, preds),
                ]
            )
        return buf.getvalue()

    @staticmethod
    def _review_cells(lab, paper, preds):
        if lab["mode"] != "model_review":
            return [""] * len(REVIEW_COLUMNS)
        run = lab["detail"].get("run", "")
        p = preds.get((run, lab["tubule_id"]), {})
        a, b = p.get("A", {}), p.get("B", {})
        return [
            run,
            paper.get(lab["tubule_id"]) or "",
            a.get("model", ""),
            a.get("value", ""),
            b.get("model", ""),
            b.get("value", ""),
        ]

    def write_csv(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.csv_text(), encoding="utf-8")
        tmp.replace(path)
        return path

    def backup(self, folder: Path, tag: str = "hourly") -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        dest = folder / f"labels_{stamp}_{tag}.sqlite"
        with self.lock:
            target = sqlite3.connect(dest)
            try:
                self.db.backup(target)
            finally:
                target.close()
        self.write_csv(folder / f"labels_{stamp}_{tag}.csv")
        return dest
