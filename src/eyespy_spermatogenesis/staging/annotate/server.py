"""Serve the annotation page on this workstation only.

Usage::

    python -m eyespy_spermatogenesis.staging.annotate.server WORKSPACE [--port 8765]

Binds to 127.0.0.1, so only this computer can open the page (decision 021).
Every label is committed to ``WORKSPACE/labels.sqlite`` as it is made. The
database and a CSV snapshot are copied to ``WORKSPACE/backups/`` at start-up,
every hour, and at shutdown. Logging out writes a CSV of all labels to
``WORKSPACE/exports/``. State-changing requests need the per-launch token
embedded in the page.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import secrets
import signal
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from eyespy_spermatogenesis.staging.annotate.store import (
    MODES,
    PAPER,
    STAGES,
    VERDICTS,
    Store,
    clean_name,
)

STATIC = resources.files("eyespy_spermatogenesis.staging.annotate") / "static"
VERSION = "0.1"


class App:
    def __init__(self, workspace: Path, backup_seconds: int = 3600):
        self.workspace = workspace.resolve()
        if not (self.workspace / "labels.sqlite").exists():
            raise SystemExit(f"No labels.sqlite in {workspace}; run prepare first")
        self.store = Store(self.workspace / "labels.sqlite")
        self.token = secrets.token_urlsafe(24)
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
                if path in ("/", "/index.html"):
                    html = (STATIC / "index.html").read_text(encoding="utf-8")
                    html = html.replace("__TOKEN__", app.token)
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
                    served = (("images",), ("paper",), ("review",))
                    if rel.parts[:1] not in served or any(
                        p in ("..", "") or p.startswith(".") for p in rel.parts
                    ):
                        return self.error_json(404, "Not found")
                    return self.file(app.workspace / rel)
                if path == "/api/config":
                    return self.json(
                        {
                            "version": VERSION,
                            "stages": STAGES,
                            "verdicts": VERDICTS,
                            "annotators": app.store.annotators(),
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
                    mode = parse_qs(url.query).get("mode", [None])[0]
                    return self.json(app.store.labels(mode))
                if path == "/api/predictions":
                    run = parse_qs(url.query).get("run", [None])[0]
                    return self.json(app.store.predictions(run))
                if path == "/api/export.csv":
                    who = parse_qs(url.query).get("who", ["manual"])[0]
                    out = app.export(who, "manual")
                    return self.send(
                        200,
                        out.read_bytes(),
                        "text/csv; charset=utf-8",
                        {
                            "Content-Disposition": f'attachment; filename="{out.name}"',
                            "X-Saved-Copy": str(out),
                        },
                    )
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


def serve(workspace: Path, port: int, open_browser: bool, backup_seconds=3600):
    app = App(workspace, backup_seconds)
    app.backup("startup")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    threading.Thread(target=app.backup_loop, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
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
    a = p.parse_args(argv)
    serve(a.workspace, a.port, not a.no_browser, a.backup_minutes * 60)


if __name__ == "__main__":
    main()
