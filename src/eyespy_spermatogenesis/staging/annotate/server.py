"""Serve the annotation page.

Usage::

    python -m eyespy_spermatogenesis.staging.annotate.server WORKSPACE [--port 8765]

Binds to 127.0.0.1. On the workstation only this computer can open it; on the
hosted server (decision 028) Cloudflare Tunnel reaches it from the same
machine, behind Cloudflare Access. ``/`` is the illustrated guide (pictures
from ``WORKSPACE/guide/``), ``/app`` the annotation page.
Every label is committed to ``WORKSPACE/labels.sqlite`` as it is made. The
database and a CSV snapshot are copied to ``WORKSPACE/backups/`` at start-up,
every hour, and at shutdown. Logging out writes a CSV of all labels to
``WORKSPACE/exports/``. State-changing requests need the token embedded in
the page. It is kept in ``WORKSPACE/token.txt`` so it survives restarts (a
deploy or checkin must not make open tabs' saves fail), and a tab whose token
is stale fetches the current one from ``/api/token`` and retries.

Open pages call ``/api/ping`` every 20 s. That tells ``scripts/sync.sh`` who has
the page open (``/api/online``) before an immediate deploy, and tells the page
when its code was replaced (the build id differs) so it can offer a reload.
Before an update the server's apply step writes ``WORKSPACE/UPDATE_NOTICE.json``;
pings then carry a countdown ("warning"), and at "closing" each page saves its
queued labels and logs out. ``--end-sessions`` ends any session left open.

Labels are anonymous on the page (decision 030): ``/api/labels`` never names who
gave a label, only whether it is the caller's own (``mine``); the only per-person
figures are counts (``/api/progress``). Names stay in the database, its backups
and exports, which only the owner sees. Another person's on-screen stage is never
sent to the page (decision 031), only that a label exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import secrets
import signal
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from eyespy_spermatogenesis.staging.annotate.hosting import guard_local_server
from eyespy_spermatogenesis.staging.annotate.store import (
    GRADING_SECTIONS,
    MODES,
    PAPER,
    STAGES,
    VERDICTS,
    Store,
    clean_name,
)

STATIC = resources.files("eyespy_spermatogenesis.staging.annotate") / "static"
VERSION = "0.2"  # 0.2 (2026-11-17): hosted updates, guest mode, anonymous labels
TOKEN_FILE = "token.txt"
ONLINE_SECONDS = 90  # seen this recently = has the page open (pings every 20 s)
NOTICE_FILE = "UPDATE_NOTICE.json"
# Detail fields that name a person; never sent to the page (decision 030).
NAMED_DETAIL = ("transcription_checked_by", "transcription_checked_at")
NAMED_KEYS = {"annotator", "annotators", *NAMED_DETAIL}
# Modes whose labels are stages, so they stay hidden on grading sheets (decision 031).
STAGE_MODES = {"staging", "model_review"}


def strip_names(obj, names: re.Pattern | None = None):
    """A copy of JSON data without any field that names a person; with
    ``names``, those names inside text become "an annotator"."""
    if isinstance(obj, dict):
        return {k: strip_names(v, names) for k, v in obj.items() if k not in NAMED_KEYS}
    if isinstance(obj, list):
        return [strip_names(v, names) for v in obj]
    if isinstance(obj, str) and names is not None:
        return names.sub(lambda m: "an annotator" + (m.group(2) or ""), obj)
    return obj


def name_pattern(names) -> re.Pattern | None:
    """Matches any of these names as a word, with an optional 's."""
    names = sorted({n for n in names if n and len(n) >= 2}, key=len, reverse=True)
    if not names:
        return None
    return re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")('s)?\b")


def when(stamp: str | None) -> float:
    """Sort key for stored times, which mix UTC and local offsets."""
    try:
        return datetime.fromisoformat(stamp).timestamp()
    except (TypeError, ValueError):
        return 0.0


def build_id() -> str:
    """Short hash of this package's files: changes whenever new code is deployed."""
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def load_token(workspace: Path) -> str:
    """The page token, made once per workspace and kept across restarts."""
    path = workspace / TOKEN_FILE
    if path.is_file():
        token = path.read_text().strip()
        if len(token) >= 32:
            return token
    token = secrets.token_urlsafe(24)
    path.write_text(token + "\n")
    path.chmod(0o600)
    return token


class App:
    def __init__(self, workspace: Path, backup_seconds: int = 3600):
        self.workspace = workspace.resolve()
        if not (self.workspace / "labels.sqlite").exists():
            raise SystemExit(f"No labels.sqlite in {workspace}; run prepare first")
        self.store = Store(self.workspace / "labels.sqlite")
        self.token = load_token(self.workspace)
        self.build = build_id()
        self.seen: dict[int, tuple[str, str, float]] = {}  # session -> who, mode, when
        self.backup_seconds = backup_seconds
        self.stop = threading.Event()
        self.last_backup = None
        info = self.workspace / "workspace.json"
        self.info = json.loads(info.read_text()) if info.exists() else {}

    # ---- backups -------------------------------------------------------
    def backup(self, tag):
        path = self.store.backup(self.workspace / "backups", tag)
        self.last_backup = datetime.now().astimezone().isoformat(timespec="seconds")
        print(f"[backup] {path}", flush=True)
        return path

    def backup_loop(self):
        while not self.stop.wait(self.backup_seconds):
            try:
                self.backup("hourly")
            except Exception as exc:  # keep serving; report loudly
                print(f"[backup] FAILED: {exc}", flush=True)

    # ---- helpers -------------------------------------------------------
    def caller(self, session_id) -> str | None:
        """Name of an open session, or None (guests, ended or missing)."""
        try:
            return self.session(session_id)["annotator"] if session_id else None
        except (PermissionError, ValueError, TypeError):
            return None

    def anonymous_labels(self, mode: str | None, session_id) -> list[dict]:
        """Labels for the page: newest first, no names, ``mine`` for the caller's.

        Another person's on-screen stage is withheld (decision 031): the row says
        only that it exists (``hidden``). On a priority scoring sample paper
        stages are withheld too. ``repeat`` marks a paper label whose author also
        staged the tubule on screen, so one person is counted once.
        """
        me = self.caller(session_id)
        grading = self.store.grading_tubules()
        rows = sorted(
            self.store.labels(mode), key=lambda r: when(r["updated_at"]), reverse=True
        )
        on_screen = {
            (r["tubule_id"], r["annotator"])
            for r in rows
            if r["mode"] == "staging" and r["origin"] == "web" and r["value"]
        }
        out = []
        for r in rows:
            mine = r["origin"] == "web" and me is not None and r["annotator"] == me
            detail = strip_names(r["detail"])
            if r["detail"].get("transcription_checked_by"):
                detail["transcription_checked"] = True
            row = {
                "tubule_id": r["tubule_id"],
                "mode": r["mode"],
                "origin": r["origin"],
                "value": r["value"],
                "detail": detail,
                "mine": mine,
            }
            on_sample = r["tubule_id"] in grading and r["mode"] in STAGE_MODES
            staged_on_screen = r["mode"] == "staging" and r["origin"] == "web"
            if not mine and (on_sample or staged_on_screen):
                row.update(value=None, detail={}, hidden=r["value"] is not None)
            if r["origin"] == PAPER and (r["tubule_id"], r["annotator"]) in on_screen:
                row["repeat"] = True
            out.append(row)
        return out

    def progress(self, mode: str, run: str | None = None) -> list[dict]:
        """Per-person label counts, in total and per stage/verdict: the only
        per-person figures the page shows (decision 030). Grading sheets count
        towards the totals but never the per-stage figures (decision 031)."""
        marks = ",".join("?" * len(GRADING_SECTIONS))
        sql = (
            f"SELECT l.annotator, l.origin, l.value, t.source_name IN ({marks}), "
            "count(*) FROM labels l "
            "JOIN tubules t ON t.tubule_id = l.tubule_id "
            "WHERE l.mode = ? AND l.value IS NOT NULL"
        )
        args: list = [*GRADING_SECTIONS, mode]
        if mode == "staging":
            sql += " AND t.recommended = 1"
        if mode == "model_review":
            sql += (
                " AND l.tubule_id IN "
                "(SELECT tubule_id FROM model_predictions WHERE run = ?)"
            )
            args.append(run)
        with self.store.lock:
            rows = self.store.db.execute(sql + " GROUP BY 1, 2, 3, 4", args).fetchall()
        people: dict[tuple, dict] = {}
        for who, origin, value, on_grading_sheet, n in rows:
            p = people.setdefault(
                (who, origin), {"name": who, "origin": origin, "n": 0, "values": {}}
            )
            p["n"] += n
            if not on_grading_sheet:
                p["values"][value] = n
        return sorted(people.values(), key=lambda p: -p["n"])

    def online(self) -> list[dict]:
        now = time.time()
        people = [
            {"annotator": who, "mode": mode, "seconds_ago": int(now - t)}
            for who, mode, t in list(self.seen.values())
            if now - t < ONLINE_SECONDS
        ]
        return sorted(people, key=lambda o: o["seconds_ago"])

    def notice(self) -> dict | None:
        """The coming update, if the server's apply step announced one."""
        try:
            n = json.loads((self.workspace / NOTICE_FILE).read_text())
            left = max(0, int(n["deadline"] - time.time()))
            return {"phase": n["phase"], "seconds_left": left}
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def session(self, session_id):
        with self.store.lock:
            row = self.store.db.execute(
                "SELECT * FROM sessions WHERE id=? AND ended_at IS NULL",
                (int(session_id),),
            ).fetchone()
        if not row:
            raise PermissionError("Session ended; log in again")
        return dict(row)

    def export(self, who: str, tag: str) -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        safe = "".join(ch if ch.isalnum() else "-" for ch in who)[:40]
        return self.store.write_csv(
            self.workspace / "exports" / f"labels_{stamp}_{safe}_{tag}.csv"
        )


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EyeSpyAnnotate/" + VERSION

        def log_message(self, fmt, *args):  # quiet: only errors
            if args and str(args[1]).startswith(("4", "5")):
                super().log_message(fmt, *args)

        def send(self, code, body: bytes, ctype: str, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj, code=200):
            self.send(code, json.dumps(obj).encode(), "application/json")

        def error_json(self, code, msg):
            self.json({"error": msg}, code)

        def file(self, path: Path, cache=True):
            if not path.is_file():
                return self.error_json(404, "Not found")
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(data)

        # ---- GET -------------------------------------------------------
        def do_GET(self):
            url = urlparse(self.path)
            path = unquote(url.path)
            try:
                if path in ("/", "/guide"):
                    # Landing page: the illustrated how-to (decision 028).
                    html = (STATIC / "guide.html").read_text(encoding="utf-8")
                    return self.send(200, html.encode(), "text/html; charset=utf-8")
                if path in ("/app", "/index.html"):
                    html = (STATIC / "index.html").read_text(encoding="utf-8")
                    html = html.replace("__TOKEN__", app.token)
                    html = html.replace("__BUILD__", app.build)
                    return self.send(200, html.encode(), "text/html; charset=utf-8")
                if path.startswith("/static/"):
                    name = path.removeprefix("/static/")
                    if "/" in name or name.startswith("."):
                        return self.error_json(404, "Not found")
                    res = STATIC / name
                    if not res.is_file():
                        return self.error_json(404, "Not found")
                    ctype = mimetypes.guess_type(name)[0] or "text/plain"
                    return self.send(200, res.read_bytes(), ctype)
                if path.startswith("/ws/"):
                    rel = Path(path.removeprefix("/ws/"))
                    served = (("images",), ("paper",), ("review",), ("guide",))
                    if rel.parts[:1] not in served or any(
                        p in ("..", "") or p.startswith(".") for p in rel.parts
                    ):
                        return self.error_json(404, "Not found")
                    f = app.workspace / rel
                    if rel.suffix == ".json" and f.is_file():
                        # e.g. review/eyespy/index.json lists training sources by name
                        data = strip_names(
                            json.loads(f.read_text(encoding="utf-8")),
                            name_pattern(app.store.all_names()),
                        )
                        return self.json(data)
                    return self.file(f, cache=rel.parts[0] != "guide")
                if path == "/api/token":
                    # Same-origin only (no CORS headers), like the token in /app.
                    return self.json({"token": app.token})
                if path == "/api/ping":
                    sid = parse_qs(url.query).get("session", [""])[0]
                    if sid.isdigit():
                        try:
                            s = app.session(sid)
                            app.seen[s["id"]] = (s["annotator"], s["mode"], time.time())
                        except PermissionError:
                            pass
                    return self.json({"build": app.build, "update": app.notice()})
                if path == "/api/online":
                    # For scripts/sync.sh on the server only: requests through the
                    # Cloudflare tunnel carry Cf-Connecting-Ip (decision 030).
                    if self.headers.get("Cf-Connecting-Ip"):
                        return self.error_json(404, "Not found")
                    return self.json(app.online())
                if path == "/api/config":
                    return self.json(
                        {
                            "version": VERSION,
                            "stages": STAGES,
                            "verdicts": VERDICTS,
                            "annotators": app.store.annotators(),
                            "grading_sections": list(GRADING_SECTIONS),
                            "workspace": str(app.workspace),
                            "info": app.info,
                            "last_backup": app.last_backup,
                            "backup_minutes": app.backup_seconds // 60,
                            "review_runs": app.store.review_runs(),
                            "overview_runs": sorted(
                                p.parent.name
                                for p in (app.workspace / "review").glob(
                                    "*/overview.json"
                                )
                            ),
                        }
                    )
                if path == "/api/session":
                    sid = parse_qs(url.query).get("id", ["0"])[0]
                    try:
                        s = app.session(sid)
                    except (PermissionError, ValueError):
                        return self.error_json(401, "Session ended; log in again")
                    return self.json(
                        {"id": s["id"], "name": s["annotator"], "mode": s["mode"]}
                    )
                if path == "/api/tubules":
                    rows = app.store.tubules()
                    for r in rows:
                        r["source_bounds"] = json.loads(r["source_bounds"])
                    return self.json(rows)
                if path == "/api/labels":
                    q = parse_qs(url.query)
                    mode = q.get("mode", [None])[0]
                    sid = q.get("session", [""])[0]
                    return self.json(app.anonymous_labels(mode, sid))
                if path == "/api/progress":
                    q = parse_qs(url.query)
                    mode = q.get("mode", ["staging"])[0]
                    if mode not in MODES:
                        return self.error_json(400, "Unknown mode")
                    return self.json(app.progress(mode, q.get("run", [None])[0]))
                if path == "/api/predictions":
                    run = parse_qs(url.query).get("run", [None])[0]
                    return self.json(app.store.predictions(run))
                return self.error_json(404, "Not found")
            except Exception as exc:
                return self.error_json(500, f"{type(exc).__name__}: {exc}")

        # ---- POST ------------------------------------------------------
        def do_POST(self):
            if self.headers.get("X-Token") != app.token:
                return self.error_json(403, "Bad token; reload the page")
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > 100_000:
                    return self.error_json(413, "Too large")
                body = json.loads(self.rfile.read(n) or b"{}")
                path = urlparse(self.path).path
                if path == "/api/login":
                    name = clean_name(body.get("name", ""))
                    if name.casefold() == "guest":
                        # reserved for guest mode, whose labels are never saved
                        return self.error_json(400, "Please use your own name")
                    mode = body.get("mode")
                    if mode not in MODES:
                        return self.error_json(400, "Choose a mode")
                    sid = app.store.start_session(name, mode)
                    return self.json({"session_id": sid, "name": name, "mode": mode})
                if path == "/api/logout":
                    s = app.session(body.get("session_id"))
                    count = app.store.session_count(s["id"])
                    out = app.export(s["annotator"], "logout")
                    app.store.end_session(s["id"], str(out))
                    app.seen.pop(s["id"], None)
                    return self.json({"export": str(out), "labels_this_session": count})
                if path == "/api/label":
                    s = app.session(body.get("session_id"))
                    tid, value = body.get("tubule_id"), body.get("value")
                    detail = None
                    if s["mode"] == "model_review":
                        run = body.get("run")
                        if run not in app.store.review_runs():
                            return self.error_json(400, "Unknown review run")
                        detail = {"run": run}
                    if value is None:
                        r = app.store.clear_label(tid, s["mode"], s["annotator"])
                    else:
                        r = app.store.set_label(
                            tid, s["mode"], s["annotator"], value, detail=detail
                        )
                    return self.json(r)
                if path == "/api/transcription":
                    s = app.session(body.get("session_id"))
                    tid, value = body.get("tubule_id"), body.get("value")
                    paper = [
                        lab
                        for lab in app.store.labels("staging")
                        if lab["tubule_id"] == tid and lab["origin"] == PAPER
                    ]
                    if not paper:
                        return self.error_json(404, "No paper label for this tubule")
                    lab = paper[0]
                    detail = {
                        "transcription_checked_by": s["annotator"],
                        "transcription_checked_at": datetime.now()
                        .astimezone()
                        .isoformat(timespec="seconds"),
                        "transcription_machine_value": lab["detail"].get(
                            "transcription_machine_value", lab["value"]
                        ),
                        "needs_check": False,
                    }
                    r = app.store.set_label(
                        tid,
                        "staging",
                        lab["annotator"],
                        value,
                        origin=PAPER,
                        actor=s["annotator"],
                        detail=detail,
                        action="transcription",
                    )
                    return self.json(r)
                return self.error_json(404, "Not found")
            except PermissionError as exc:
                return self.error_json(401, str(exc))
            except (ValueError, KeyError, TypeError) as exc:
                return self.error_json(400, f"{type(exc).__name__}: {exc}")
            except Exception as exc:
                return self.error_json(500, f"{type(exc).__name__}: {exc}")

    return Handler


def end_sessions(workspace: Path) -> int:
    """End every open session (before an update), with one CSV export."""
    store = Store(workspace / "labels.sqlite")
    try:
        with store.lock:
            ids = [
                row[0]
                for row in store.db.execute(
                    "SELECT id FROM sessions WHERE ended_at IS NULL"
                )
            ]
        if ids:
            stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            out = store.write_csv(
                workspace / "exports" / f"labels_{stamp}_before-update.csv"
            )
            for sid in ids:
                store.end_session(sid, str(out))
        return len(ids)
    finally:
        store.close()


def serve(workspace: Path, port: int, open_browser: bool, backup_seconds=3600):
    guard_local_server(workspace)
    app = App(workspace, backup_seconds)
    app.backup("startup")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    threading.Thread(target=app.backup_loop, daemon=True).start()
    url = f"http://127.0.0.1:{port}/app"
    print(f"Annotation page: {url}  (Ctrl+C to stop)", flush=True)

    def shutdown(*_):
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop.set()
        httpd.server_close()
        app.backup("shutdown")
        app.store.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("workspace", type=Path)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--backup-minutes", type=int, default=60)
    p.add_argument(
        "--end-sessions",
        action="store_true",
        help="end every open session and exit (run before an update)",
    )
    a = p.parse_args(argv)
    if a.end_sessions:
        n = end_sessions(a.workspace.resolve())
        print(f"Ended {n} open session(s).", flush=True)
        return
    serve(a.workspace, a.port, not a.no_browser, a.backup_minutes * 60)


if __name__ == "__main__":
    main()
