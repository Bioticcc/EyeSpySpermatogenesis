"""Keep the hosted annotation page and this workstation in step (decision 028).

Usage (normally through ``scripts/sync.sh``)::

    python -m eyespy_spermatogenesis.staging.annotate.hosting COMMAND [--config FILE]

Once hosted, the server's ``labels.sqlite`` is the live database. This
workstation keeps a copy that ``pull`` refreshes; it is read-only here.

==========  ==============================================================
setup       One time: upload the code, workspace and database; install the
            service. Refuses if the server already has a database.
deploy      Queue the code for tonight: a timer on the server applies it at
            ``update_at`` (default 22:00 America/Los_Angeles), so this
            workstation need not be on. If anyone has the page open they get
            a ``warn_seconds`` countdown, then their pages save and log out;
            sessions left open are ended with a CSV export; then the new code
            goes in. ``--now`` does that at once if nobody is online
            (``--force``: even if they are); ``--cancel`` drops it.
status      Is the page running, how many labels, disk space, checkout.
pull        Copy the live database, backups and exports here. Safe while
            people annotate. Run it whenever you want fresh labels.
checkout    Turn the page OFF and pull, so a local command may change the
            database (prepare, paper_import, eyespy review, ...).
checkin     Upload the changed database and files, turn the page back ON.
            Refuses if the server's database changed since checkout.
            ``--discard`` instead throws local changes away and restarts.
push-files  Upload images/, paper/, review/, guide/ and workspace.json
            without touching the database (safe while the page runs).
==========  ==============================================================

``WORKSPACE/HOSTED.json`` marks a hosted workspace (the local page then refuses
to start, so no label is made here and lost); ``WORKSPACE/CHECKOUT.json``
marks a checkout in progress.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[4]
# Folders the page shows; uploaded from here. backups/ and exports/ are
# written on the server and only ever come down.
FILES_UP = ["images", "paper", "review", "guide", "workspace.json"]
FILES_DOWN = ["backups", "exports"]
HOSTED = "HOSTED.json"
# One shared ssh connection for every command and copy (many quick logins in a
# row can be throttled by the server).
SSH_OPTS = [
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=~/.ssh/eyespy-sync-%C",
    "-o", "ControlPersist=120",
    "-o", "ServerAliveInterval=30",
]  # fmt: skip
CHECKOUT = "CHECKOUT.json"


@dataclass
class Config:
    host: str  # ssh target, e.g. ubuntu@203.0.113.10; "local" for tests
    remote_dir: str = "/srv/eyespy"
    workspace: Path = Path("outputs/staging-annotation-2026-11-06-01")
    service: str = "eyespy-annotate"
    service_user: str = "eyespy"  # runs the page; "" = leave ownership alone
    port: int = 8765
    keep_days: int = 14  # remote backups older than this go once copied here
    update_at: str = "22:00"  # queued code updates go live at this time...
    timezone: str = "America/Los_Angeles"  # ...in this zone (Washington state)
    warn_seconds: int = 300  # countdown shown to people online before an update
    grace_seconds: int = 60  # then at most this long for their pages to log out
    install_service: bool = True
    service_control: str = ""  # tests only: shell template with {action}

    @classmethod
    def load(cls, path: Path) -> Config:
        if not path.exists():
            raise SystemExit(
                f"No {path}. Copy the example in docs/HOSTING.md and fill in the host."
            )
        data = tomllib.loads(path.read_text())
        try:
            cfg = cls(**data)
        except TypeError as exc:
            raise SystemExit(f"{path}: {exc}") from None
        try:
            ZoneInfo(cfg.timezone)
            datetime.strptime(cfg.update_at, "%H:%M")
        except Exception:
            raise SystemExit(
                f"{path}: update_at must be HH:MM and timezone a zone name "
                f"such as America/Los_Angeles"
            ) from None
        ws = Path(cfg.workspace)
        cfg.workspace = (ws if ws.is_absolute() else REPO / ws).resolve()
        return cfg

    @property
    def remote_ws(self) -> str:
        return f"{self.remote_dir}/workspace"

    @property
    def queue_dir(self) -> str:
        return f"{self.remote_dir}/next"

    @property
    def updater(self) -> str:
        return f"{self.service}-update"


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def say(msg: str):
    print(msg, flush=True)


class Remote:
    """Runs commands and copies files over ssh/rsync (or locally for tests)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.local = cfg.host == "local"
        # A non-root login (ubuntu@ on AWS Lightsail) works through passwordless
        # sudo, the cloud images' default.
        self.sudo = not self.local and not cfg.host.startswith("root@")

    def run(self, script: str, check=True) -> subprocess.CompletedProcess:
        cmd = ["bash", "-c", script]
        if not self.local:
            shell = "sudo bash -c " if self.sudo else "bash -c "
            cmd = ["ssh", *SSH_OPTS, self.cfg.host, shell + shlex.quote(script)]
        r = self._retry(cmd)
        if check and r.returncode:
            raise SystemExit(f"Server command failed:\n{script}\n{r.stderr.strip()}")
        return r

    def _retry(self, cmd: list[str], tries: int = 5) -> subprocess.CompletedProcess:
        """Run, retrying when the ssh connection itself fails (exit 255 for ssh,
        12 or 255 for rsync), e.g. an intermittent "Connection reset by peer"."""
        for attempt in range(tries):
            r = subprocess.run(cmd, capture_output=True, text=True)
            dropped = r.returncode in (12, 255) and (
                "Connection reset" in r.stderr
                or "Connection closed" in r.stderr
                or "connection unexpectedly closed" in r.stderr
                or "Connection timed out" in r.stderr
            )
            if self.local or not dropped or attempt == tries - 1:
                return r
            time.sleep(3 * (attempt + 1))
        return r

    def out(self, script: str) -> str:
        return self.run(script).stdout.strip()

    def where(self, path: str) -> str:
        return path if self.local else f"{self.cfg.host}:{path}"

    def rsync(self, *args: str):
        extra = ["--rsync-path=sudo rsync"] if self.sudo else []
        if not self.local:
            extra.append("--rsh=" + shlex.join(["ssh", *SSH_OPTS]))
        r = self._retry(["rsync", "-a", *extra, *args])
        if r.returncode:
            raise SystemExit(f"rsync failed: {' '.join(args)}\n{r.stderr.strip()}")

    def control(self, action: str, check=True) -> bool:
        if self.cfg.service_control:
            script = self.cfg.service_control.format(action=action)
        else:
            script = f"systemctl {action} {shlex.quote(self.cfg.service)}"
        return self.run(script, check=check).returncode == 0

    def active(self) -> bool:
        return self.control("is-active", check=False)

    def python(self, code: str, *args: str) -> str:
        """Server-side Python, as the service user so every database file it
        touches stays owned by the page."""
        quoted = " ".join(shlex.quote(a) for a in args)
        user = self.cfg.service_user
        prefix = f"runuser -u {shlex.quote(user)} -- " if user else ""
        return self.out(f"{prefix}python3 -c {shlex.quote(code)} {quoted}")

    def chown(self):
        if self.cfg.service_user:
            user = shlex.quote(self.cfg.service_user)
            ws, tmp = self.cfg.remote_ws, f"{self.cfg.remote_dir}/tmp"
            self.run(f"chown -R {user}: {shlex.quote(ws)} {shlex.quote(tmp)}")


# Python run on the server (stdlib only).
SNAPSHOT = """
import sqlite3, sys
src = sqlite3.connect(sys.argv[1])
dst = sqlite3.connect(sys.argv[2]); src.backup(dst); dst.close(); src.close()
"""
DIGEST = """
import hashlib, pathlib, sys
h = hashlib.sha256()
for suffix in ('', '-wal'):
    p = pathlib.Path(sys.argv[1] + suffix)
    h.update(p.read_bytes() if p.exists() and p.stat().st_size else b'')
print(h.hexdigest())
"""
COUNTS = """
import json, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
q = lambda s: db.execute(s).fetchone()[0]
n = "SELECT count(*) FROM labels WHERE value IS NOT NULL AND origin="
print(json.dumps({
  'labels_on_screen': q(n + "'web'"),
  'labels_paper': q(n + "'paper'"),
  'annotators': q("SELECT count(DISTINCT annotator) FROM labels WHERE origin='web'"),
  'last_change': q('SELECT max(at) FROM events'),
}))
"""
# Who has the page open (the page pings every 20 s); [] if it is not running.
ONLINE = """
import sys, urllib.request
try:
    print(urllib.request.urlopen(sys.argv[1], timeout=5).read().decode())
except Exception:
    print('[]')
"""


def snapshot_local(src: Path, dst: Path):
    """Consistent copy of a SQLite database (WAL included), via the backup API."""
    a = sqlite3.connect(src)
    b = sqlite3.connect(dst)
    a.backup(b)
    b.close()
    a.close()


def check_integrity(path: Path):
    db = sqlite3.connect(path)
    ok = db.execute("PRAGMA integrity_check").fetchone()[0]
    db.close()
    if ok != "ok":
        raise SystemExit(f"{path} failed its integrity check: {ok}")


def counts_local(path: Path) -> dict:
    db = sqlite3.connect(path)
    q = lambda s: db.execute(s).fetchone()[0]  # noqa: E731
    out = {
        "labels_on_screen": q(
            "SELECT count(*) FROM labels WHERE origin='web' AND value IS NOT NULL"
        ),
        "last_change": q("SELECT max(at) FROM events"),
    }
    db.close()
    return out


def read_marker(cfg: Config, name: str) -> dict | None:
    p = cfg.workspace / name
    return json.loads(p.read_text()) if p.exists() else None


def write_marker(cfg: Config, name: str, data: dict):
    (cfg.workspace / name).write_text(json.dumps(data, indent=2) + "\n")


# ---- commands ----------------------------------------------------------
def deploy_code(cfg: Config, r: Remote, dest: str | None = None):
    """Upload src/ to ``dest`` (default: the live code in app/)."""
    dest = dest or f"{cfg.remote_dir}/app"
    r.run(f"mkdir -p {shlex.quote(dest)}")
    r.rsync(
        "--delete",
        "--exclude=__pycache__",
        str(REPO / "src") + "/",
        r.where(f"{dest}/src/"),
    )


def code_version() -> dict:
    """The commit being uploaded, and whether src/ has uncommitted changes."""

    def git(*args):
        cmd = ["git", "-C", str(REPO), *args]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""

    return {
        "commit": git("rev-parse", "--short", "HEAD") or "unknown",
        "uncommitted": bool(git("status", "--porcelain", "--", "src")),
        "at": stamp(),
    }


def write_remote_json(r: Remote, path: str, data: dict):
    body = json.dumps(data, indent=2)
    r.run(f"cat > {shlex.quote(path)} <<'JSON'\n{body}\nJSON")


def read_remote_json(r: Remote, path: str) -> dict | None:
    out = r.out(f"cat {shlex.quote(path)} 2>/dev/null || true")
    return json.loads(out) if out else None


APPLY = r'''#!/usr/bin/env python3
"""Installed by scripts/sync.sh deploy (decision 028); run by the update timer.

Puts a queued code update live. People with the page open get a countdown
first, then their pages save and log out; sessions still open are ended (with
a CSV export) once the page has stopped. A stopped (checked-out) page only
gets the new code: checkin starts it. On any failure the page is started again.
"""
import json, os, subprocess, sys, time, urllib.request
from datetime import datetime
from pathlib import Path

CFG = json.loads(r"""__CFG__""")
D = Path(CFG["remote_dir"])
WS, QUEUE, APP = D / "workspace", D / "next", D / "app"
NOTICE = WS / "UPDATE_NOTICE.json"


def say(msg):
    print(datetime.now().strftime("%H:%M:%S"), msg, flush=True)


def control(action):
    cmd = CFG["control"].format(action=action)
    return subprocess.run(cmd, shell=True).returncode == 0


def online():
    url = f"http://127.0.0.1:{CFG['port']}/api/online"
    try:
        return json.loads(urllib.request.urlopen(url, timeout=5).read())
    except Exception:
        return []


def notice(phase, seconds):
    tmp = NOTICE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"phase": phase, "deadline": time.time() + seconds}))
    os.chmod(tmp, 0o644)
    tmp.replace(NOTICE)


def end_sessions():
    module = "eyespy_spermatogenesis.staging.annotate.server"
    cmd = [sys.executable, "-m", module, "--end-sessions", str(WS)]
    if CFG["service_user"]:
        cmd = ["runuser", "-u", CFG["service_user"], "--", *cmd]
    # The queued (new) code: the live code may predate --end-sessions.
    env = dict(os.environ, PYTHONPATH=str(QUEUE / "src"))
    subprocess.run(cmd, env=env, check=True)


def main():
    queued = QUEUE / "QUEUED.json"
    if not queued.exists():
        say("No queued update")
        return
    info = json.loads(queued.read_text())
    running = control("is-active")
    try:
        if running:
            people = online()
            if people:
                warn = int(info.get("warn_seconds", CFG["warn_seconds"]))
                say(f"Warning {len(people)} people online; update in {warn} s")
                notice("warning", warn)
                time.sleep(warn)
                notice("closing", CFG["grace_seconds"])
                end = time.time() + CFG["grace_seconds"]
                while online() and time.time() < end:
                    time.sleep(2)
                say(f"Pages saved and logged out; {len(online())} did not answer")
            control("stop")
            end_sessions()
        subprocess.run(
            ["rsync", "-a", "--delete", "--exclude=__pycache__",
             f"{QUEUE}/src/", f"{APP}/src/"],
            check=True,
        )
        info["applied"] = datetime.now().astimezone().isoformat(timespec="seconds")
        (APP / "DEPLOYED.json").write_text(json.dumps(info, indent=2) + "\n")
        subprocess.run(["rm", "-rf", str(QUEUE)], check=True)
        say(f"Applied {info.get('commit')}")
    finally:
        NOTICE.unlink(missing_ok=True)
        if running:
            control("start")


main()
'''


def apply_script(cfg: Config) -> str:
    control = cfg.service_control or f"systemctl {{action}} {shlex.quote(cfg.service)}"
    settings = {
        "remote_dir": cfg.remote_dir,
        "port": cfg.port,
        "service_user": cfg.service_user,
        "control": control,
        "warn_seconds": cfg.warn_seconds,
        "grace_seconds": cfg.grace_seconds,
    }
    return APPLY.replace("__CFG__", json.dumps(settings))


def updater_script(cfg: Config) -> str:
    return f"{cfg.remote_dir}/bin/apply-update.py"


def install_updater(cfg: Config, r: Remote):
    """The apply script and (on a real server) a systemd timer that runs it."""
    q = shlex.quote(updater_script(cfg))
    r.run(
        f"mkdir -p {shlex.quote(cfg.remote_dir)}/bin && cat > {q} <<'PY'\n"
        f"{apply_script(cfg)}PY\nchmod 755 {q}"
    )
    if not cfg.install_service:
        return
    when = f"*-*-* {cfg.update_at}:00 {cfg.timezone}"
    if r.run(f"systemd-analyze calendar {shlex.quote(when)}", check=False).returncode:
        raise SystemExit(f"The server's systemd does not accept the time {when!r}.")
    py = r.out("command -v python3")
    service = f"""[Unit]
Description=Apply a queued EyeSpy annotation page update

[Service]
Type=oneshot
ExecStart={py} {updater_script(cfg)}
TimeoutStartSec=30min
"""
    timer = f"""[Unit]
Description=Apply queued EyeSpy page updates ({cfg.update_at} {cfg.timezone})

[Timer]
OnCalendar={when}

[Install]
WantedBy=timers.target
"""
    base = f"/etc/systemd/system/{cfg.updater}"
    unit = shlex.quote(cfg.updater)
    r.run(
        f"cat > {shlex.quote(base + '.service')} <<'UNIT'\n{service}UNIT\n"
        f"cat > {shlex.quote(base + '.timer')} <<'UNIT'\n{timer}UNIT\n"
        f"systemctl daemon-reload && systemctl enable --now {unit}.timer"
    )


def apply_queued(cfg: Config, r: Remote):
    """Run the apply step now, as the timer would (waits until it is done)."""
    if r.local:
        r.run(f"{shlex.quote(sys.executable)} {shlex.quote(updater_script(cfg))}")
        return
    start = f"systemctl start {shlex.quote(cfg.updater)}.service"
    if r.run(start, check=False).returncode:
        raise SystemExit(
            "The update failed; the page was restarted on the old code. "
            f"See `journalctl -u {cfg.updater} -n 50` on the server."
        )


def next_update(cfg: Config) -> datetime:
    tz = ZoneInfo(cfg.timezone)
    now = datetime.now(tz)
    h, m = (int(x) for x in cfg.update_at.split(":"))
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return t if t > now else (t + timedelta(days=1)).replace(hour=h, minute=m)


def online(cfg: Config, r: Remote) -> list[dict]:
    """People with the page open in the last few minutes (from the page's pings)."""
    out = r.python(ONLINE, f"http://127.0.0.1:{cfg.port}/api/online")
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return []


def describe_online(people: list[dict]) -> str:
    return ", ".join(
        f"{p['annotator']} ({p['mode']}, seen {p['seconds_ago']} s ago)" for p in people
    )


def push_files(cfg: Config, r: Remote):
    r.run(f"mkdir -p {shlex.quote(cfg.remote_ws)} {shlex.quote(cfg.remote_dir)}/tmp")
    for name in FILES_UP:
        src = cfg.workspace / name
        if src.exists():
            say(f"  uploading {name}")
            r.rsync(
                str(src) + ("/" if src.is_dir() else ""),
                r.where(f"{cfg.remote_ws}/{name}" + ("/" if src.is_dir() else "")),
            )
    r.chown()


def upload_db(cfg: Config, r: Remote, tmp: Path):
    """Local database to the server, keeping a copy of the one it replaces."""
    local = cfg.workspace / "labels.sqlite"
    check_integrity(local)
    snap = tmp / "upload.sqlite"
    snapshot_local(local, snap)
    remote_tmp = f"{cfg.remote_dir}/tmp/upload.sqlite"
    r.run(
        f"mkdir -p {shlex.quote(cfg.remote_dir)}/tmp"
        f" {shlex.quote(cfg.remote_ws)}/backups"
    )
    r.rsync(str(snap), r.where(remote_tmp))
    db = f"{cfg.remote_ws}/labels.sqlite"
    keep = f"{cfg.remote_ws}/backups/labels_{stamp()}_before_checkin.sqlite"
    q_db, q_keep, q_tmp = shlex.quote(db), shlex.quote(keep), shlex.quote(remote_tmp)
    r.run(
        f"set -e; if [ -e {q_db} ]; then cp {q_db} {q_keep}; fi;"
        f" mv {q_tmp} {q_db}; rm -f {q_db}-wal {q_db}-shm"
    )
    r.chown()


def ensure_user(cfg: Config, r: Remote):
    """The unprivileged user that runs the page (needed before any chown)."""
    if cfg.service_user:
        u = shlex.quote(cfg.service_user)
        r.run(
            f"id -u {u} >/dev/null 2>&1 || useradd --system --home-dir "
            f"{shlex.quote(cfg.remote_dir)} --shell /usr/sbin/nologin {u}"
        )


def install_service(cfg: Config, r: Remote):
    py = r.out("command -v python3")
    user = cfg.service_user or "root"
    module = "eyespy_spermatogenesis.staging.annotate.server"
    args = f"{cfg.remote_ws} --port {cfg.port} --no-browser"
    unit = f"""[Unit]
Description=EyeSpy staging annotation page
After=network.target

[Service]
User={user}
WorkingDirectory={cfg.remote_dir}
Environment=PYTHONPATH={cfg.remote_dir}/app/src
ExecStart={py} -m {module} {args}
Restart=on-failure
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
"""
    path = f"/etc/systemd/system/{cfg.service}.service"
    r.run(f"cat > {shlex.quote(path)} <<'UNIT'\n{unit}UNIT\nsystemctl daemon-reload")
    r.chown()
    r.run(f"systemctl enable {shlex.quote(cfg.service)}")


def cmd_setup(cfg: Config, r: Remote):
    db = f"{cfg.remote_ws}/labels.sqlite"
    if r.run(f"test -e {shlex.quote(db)}", check=False).returncode == 0:
        raise SystemExit(
            "The server already has a database. Use pull / checkout / checkin; "
            "setup never overwrites it."
        )
    version = r.out(
        "python3 -c 'import sys; print(sys.version_info >= (3, 11))' 2>/dev/null"
        " || echo missing"
    )
    if version != "True":
        raise SystemExit("The server needs Python 3.11 or newer (python3).")
    ensure_user(cfg, r)
    say("Uploading code")
    deploy_code(cfg, r)
    say("Uploading page files (the first time takes a while: about 4 GB)")
    push_files(cfg, r)
    say("Uploading the database")
    with tempfile.TemporaryDirectory() as tmp:
        upload_db(cfg, r, Path(tmp))
    if cfg.install_service:
        say("Installing the service")
        install_service(cfg, r)
    r.control("start")
    write_marker(
        cfg, HOSTED, {"host": cfg.host, "remote_dir": cfg.remote_dir, "since": stamp()}
    )
    say("The page is running on the server. From now on its database is the live one.")


def cmd_deploy(cfg: Config, r: Remote, now=False, force=False, cancel=False):
    if cancel:
        r.run(f"rm -rf {shlex.quote(cfg.queue_dir)}")
        say("Queued update dropped; the live code is unchanged.")
        return
    version = {**code_version(), "warn_seconds": cfg.warn_seconds}
    if version["uncommitted"]:
        say(
            "Note: src/ has uncommitted changes and they are included. Commit them "
            "so you know which code is live."
        )
    install_updater(cfg, r)
    deploy_code(cfg, r, cfg.queue_dir)
    write_remote_json(r, f"{cfg.queue_dir}/QUEUED.json", version)
    when = next_update(cfg)
    hours = (when - datetime.now(when.tzinfo)).total_seconds() / 3600
    tonight = f"{when:%a %H:%M %Z} ({cfg.timezone}, in {hours:.1f} h)"
    if not now:
        say(
            f"Queued {version['commit']}. It goes live {tonight}, after a "
            f"{cfg.warn_seconds // 60}-minute warning if anyone is online. A later "
            "deploy replaces it; `deploy --cancel` drops it; `deploy --now` applies "
            "it at once."
        )
        return
    people = online(cfg, r)
    if people and not force:
        raise SystemExit(
            f"Not applied now: the page is open for {describe_online(people)}. It "
            f"stays queued for {tonight}. `deploy --now --force` applies it now "
            f"after a {cfg.warn_seconds // 60}-minute warning."
        )
    if people:
        say(
            f"Warning {describe_online(people)}: they are logged out in "
            f"{cfg.warn_seconds // 60} min, then the update goes in. Waiting..."
        )
    apply_queued(cfg, r)
    live = read_remote_json(r, f"{cfg.remote_dir}/app/DEPLOYED.json") or {}
    if live.get("at") != version["at"]:
        raise SystemExit("The update did not go in; see the server's journal.")
    off = " The page stays off until checkin." if read_marker(cfg, CHECKOUT) else ""
    say(f"Deployed {version['commit']}.{off}")


def pull(cfg: Config, r: Remote, tmp: Path):
    """Live database (consistent snapshot) and new backups/exports to here."""
    db = f"{cfg.remote_ws}/labels.sqlite"
    remote_snap = f"{cfg.remote_dir}/tmp/pull.sqlite"
    r.run(f"mkdir -p {shlex.quote(cfg.remote_dir)}/tmp")
    r.chown()
    r.python(SNAPSHOT, db, remote_snap)
    snap = tmp / "pull.sqlite"
    r.rsync(r.where(remote_snap), str(snap))
    check_integrity(snap)
    local = cfg.workspace / "labels.sqlite"
    (cfg.workspace / "backups").mkdir(exist_ok=True)
    if local.exists():
        snapshot_local(
            local, cfg.workspace / "backups" / f"labels_{stamp()}_before_pull.sqlite"
        )
    a = sqlite3.connect(snap)
    b = sqlite3.connect(local)
    a.backup(b)
    b.close()
    a.close()
    for name in FILES_DOWN:
        r.run(f"mkdir -p {shlex.quote(cfg.remote_ws)}/{name}")
        (cfg.workspace / name).mkdir(exist_ok=True)
        r.rsync(r.where(f"{cfg.remote_ws}/{name}/"), str(cfg.workspace / name) + "/")
    prune_remote_backups(cfg, r)
    c = counts_local(local)
    say(
        f"Pulled: {c['labels_on_screen']} on-screen labels, last change "
        f"{c['last_change']}."
    )


def prune_remote_backups(cfg: Config, r: Remote):
    """Delete server backups older than keep_days that are already copied here."""
    folder = f"{cfg.remote_ws}/backups"
    listing = r.out(
        f"find {shlex.quote(folder)} -maxdepth 1 -type f -mtime +{int(cfg.keep_days)} "
        "-printf '%f %s\\n'"
    )
    gone = []
    for line in listing.splitlines():
        name, size = line.rsplit(" ", 1)
        here = cfg.workspace / "backups" / name
        if here.exists() and here.stat().st_size == int(size):
            gone.append(f"{folder}/{name}")
    if gone:
        r.run("rm -f " + " ".join(shlex.quote(g) for g in gone))
        say(
            f"Removed {len(gone)} server backups older than {cfg.keep_days} days"
            " (copies are here)."
        )


def remote_digest(cfg: Config, r: Remote) -> str:
    return r.python(DIGEST, f"{cfg.remote_ws}/labels.sqlite")


def cmd_pull(cfg: Config, r: Remote):
    if read_marker(cfg, CHECKOUT):
        raise SystemExit(
            "Checked out: a pull would overwrite your local changes. Run checkin "
            "(or checkin --discard) first."
        )
    with tempfile.TemporaryDirectory() as tmp:
        pull(cfg, r, Path(tmp))


def cmd_checkout(cfg: Config, r: Remote):
    if read_marker(cfg, CHECKOUT):
        raise SystemExit("Already checked out. Run checkin when you are done.")
    say("Turning the hosted page OFF")
    r.control("stop")
    # Marker first: if the pull fails, `checkin --discard` turns the page back on.
    write_marker(cfg, CHECKOUT, {"at": stamp(), "remote_sha256": remote_digest(cfg, r)})
    with tempfile.TemporaryDirectory() as tmp:
        pull(cfg, r, Path(tmp))
    say(
        "Checked out. The hosted page is OFF until you run checkin. Run your "
        "commands on this workspace now."
    )


def cmd_checkin(cfg: Config, r: Remote, discard=False):
    mark = read_marker(cfg, CHECKOUT)
    if not mark:
        raise SystemExit("Not checked out; nothing to check in.")
    if discard:
        r.control("start")
        (cfg.workspace / CHECKOUT).unlink()
        say("Local changes discarded; the hosted page is ON. Pulling the live copy.")
        cmd_pull(cfg, r)
        return
    if r.active():
        raise SystemExit(
            "The hosted page is running again; someone started it. Not uploading."
        )
    if remote_digest(cfg, r) != mark["remote_sha256"]:
        raise SystemExit(
            "The server's database changed after checkout. Not uploading, so no label "
            "is lost. Run `checkin --discard`, then checkout again and redo your step."
        )
    say("Uploading page files")
    push_files(cfg, r)
    say("Uploading the database")
    with tempfile.TemporaryDirectory() as tmp:
        upload_db(cfg, r, Path(tmp))
    r.control("start")
    (cfg.workspace / CHECKOUT).unlink()
    say("Checked in. The hosted page is ON.")


def cmd_push_files(cfg: Config, r: Remote):
    push_files(cfg, r)
    say("Files uploaded. Reload the page to see them.")


def cmd_status(cfg: Config, r: Remote):
    say(f"Server: {cfg.host}  page {'RUNNING' if r.active() else 'OFF'}")
    try:
        c = json.loads(r.python(COUNTS, f"{cfg.remote_ws}/labels.sqlite"))
        say(
            f"Live database: {c['labels_on_screen']} on-screen labels by "
            f"{c['annotators']} people, {c['labels_paper']} paper; last change "
            f"{c['last_change']}"
        )
    except SystemExit as exc:
        say(f"Live database: not readable ({exc})")
    say(
        "Disk: "
        + r.out(
            f"df -h --output=used,avail,pcent {shlex.quote(cfg.remote_dir)} | tail -1"
        )
    )
    mark = read_marker(cfg, CHECKOUT)
    say(f"Checkout: since {mark['at']}" if mark else "Checkout: none")
    people = online(cfg, r)
    say(f"Online now: {describe_online(people)}" if people else "Online now: nobody")
    live = read_remote_json(r, f"{cfg.remote_dir}/app/DEPLOYED.json")
    if live:
        say(f"Live code: {live['commit']} (deployed {live['at']})")
    queued = read_remote_json(r, f"{cfg.queue_dir}/QUEUED.json")
    if queued:
        when = next_update(cfg)
        say(f"Queued update: {queued['commit']}, goes live {when:%a %H:%M %Z}")
    else:
        say("Queued update: none")
    local = cfg.workspace / "labels.sqlite"
    if local.exists():
        c = counts_local(local)
        say(
            f"Local copy: {c['labels_on_screen']} on-screen labels, last change"
            f" {c['last_change']}"
        )


def guard_local_server(workspace: Path):
    """Stop the local page from labelling into a copy the next pull overwrites."""
    hosted = workspace / HOSTED
    if hosted.exists() and not (workspace / CHECKOUT).exists():
        info = json.loads(hosted.read_text())
        raise SystemExit(
            f"This workspace is hosted on {info.get('host')} (decision 028). Labels "
            "made here would be overwritten by the next pull. Use the hosted page, or "
            "run `scripts/sync.sh checkout` first."
        )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "command",
        choices=[
            "setup",
            "deploy",
            "status",
            "pull",
            "checkout",
            "checkin",
            "push-files",
        ],
    )
    p.add_argument("--config", type=Path, default=REPO / "hosting.toml")
    p.add_argument(
        "--discard", action="store_true", help="checkin: drop local changes, restart"
    )
    p.add_argument("--now", action="store_true", help="deploy: apply at once")
    p.add_argument(
        "--force", action="store_true", help="deploy --now: even if people are online"
    )
    p.add_argument(
        "--cancel", action="store_true", help="deploy: drop the queued update"
    )
    a = p.parse_args(argv)
    if not shutil.which("rsync"):
        raise SystemExit("rsync is not installed")
    cfg = Config.load(a.config)
    r = Remote(cfg)
    if a.force and not a.now:
        raise SystemExit("--force only goes with deploy --now.")
    if a.command == "checkin":
        return cmd_checkin(cfg, r, a.discard)
    if a.command == "deploy":
        return cmd_deploy(cfg, r, now=a.now, force=a.force, cancel=a.cancel)
    {
        "setup": cmd_setup,
        "status": cmd_status,
        "pull": cmd_pull,
        "checkout": cmd_checkout,
        "push-files": cmd_push_files,
    }[a.command](cfg, r)


if __name__ == "__main__":
    sys.exit(main())
