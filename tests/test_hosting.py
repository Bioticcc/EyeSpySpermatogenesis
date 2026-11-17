"""Hosted-page sync (decision 028), run against a "server" folder on this machine."""

import json
import os
import shutil
import socket
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from eyespy_spermatogenesis.staging.annotate import hosting
from eyespy_spermatogenesis.staging.annotate.server import App, make_handler
from eyespy_spermatogenesis.staging.annotate.store import Store

pytestmark = pytest.mark.skipif(not shutil.which("rsync"), reason="needs rsync")
TID = "src/series-0/candidate-1"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def tubule(i):
    return {
        "tubule_id": f"src/series-0/candidate-{i}",
        "seq": i,
        "source_folder": "source_01",
        "source_name": "a.tif",
        "source_sha256": "x",
        "label": i,
        "recommended": 1,
        "flags": "",
        "source_bounds": json.dumps([0, 0, 10, 10]),
        "image": f"images/source_01/c_{i:05d}.jpg",
        "outline": f"images/source_01/o_{i:05d}.png",
        "display_scale": 1.0,
    }


@pytest.fixture
def site(tmp_path):
    ws = tmp_path / "local"
    ws.mkdir()
    s = Store(ws / "labels.sqlite")
    s.add_tubules([tubule(1), tubule(2)])
    s.close()
    (ws / "images" / "source_01").mkdir(parents=True)
    (ws / "images" / "source_01" / "c_00001.jpg").write_bytes(b"jpg")
    state = tmp_path / "running"
    cfg = hosting.Config(
        host="local",
        remote_dir=str(tmp_path / "server"),
        workspace=ws,
        service_user="",
        install_service=False,
        port=free_port(),  # nothing listens: nobody online unless a test serves
        warn_seconds=2,
        grace_seconds=3,
        # A stand-in for systemctl: the page "runs" while the file exists.
        service_control=(
            f"case {{action}} in start|restart) touch {state};; stop) rm -f {state};;"
            f" is-active) test -e {state};; esac"
        ),
    )
    r = hosting.Remote(cfg)
    hosting.cmd_setup(cfg, r)
    return cfg, r, tmp_path / "server" / "workspace"


def web_labels(db):
    s = Store(db)
    out = sorted(
        (lab["annotator"], lab["tubule_id"][-1], lab["value"])
        for lab in s.labels("staging")
    )
    s.close()
    return out


def test_setup_uploads_and_refuses_twice(site):
    cfg, r, remote = site
    assert (remote / "images" / "source_01" / "c_00001.jpg").read_bytes() == b"jpg"
    assert (remote / "labels.sqlite").exists()
    assert r.active()
    assert (cfg.workspace / hosting.HOSTED).exists()
    with pytest.raises(SystemExit, match="already has a database"):
        hosting.cmd_setup(cfg, r)


def test_pull_brings_live_labels_and_keeps_a_local_backup(site):
    cfg, r, remote = site
    s = Store(remote / "labels.sqlite")  # an annotator labels on the hosted page
    s.set_label(TID, "staging", "Ann", "IV")
    hosting.cmd_pull(cfg, r)
    assert web_labels(cfg.workspace / "labels.sqlite") == [("Ann", "1", "IV")]
    assert list((cfg.workspace / "backups").glob("*_before_pull.sqlite"))
    s.close()


def test_checkout_checkin_round_trip(site):
    cfg, r, remote = site
    s = Store(remote / "labels.sqlite")
    s.set_label(TID, "staging", "Ann", "IV")
    s.close()
    hosting.cmd_checkout(cfg, r)
    assert not r.active()
    with pytest.raises(SystemExit, match="Checked out"):
        hosting.cmd_pull(cfg, r)
    s = Store(cfg.workspace / "labels.sqlite")  # a local command writes
    s.set_label("src/series-0/candidate-2", "staging", "paper-import", "V")
    s.close()
    (cfg.workspace / "review").mkdir()
    (cfg.workspace / "review" / "x.json").write_text("{}")
    hosting.cmd_checkin(cfg, r)
    assert r.active()
    assert not (cfg.workspace / hosting.CHECKOUT).exists()
    assert web_labels(remote / "labels.sqlite") == [
        ("Ann", "1", "IV"),
        ("paper-import", "2", "V"),
    ]
    assert (remote / "review" / "x.json").exists()
    assert list((remote / "backups").glob("*_before_checkin.sqlite"))


def test_checkin_refuses_when_server_changed(site):
    cfg, r, remote = site
    hosting.cmd_checkout(cfg, r)
    s = Store(remote / "labels.sqlite")  # someone wrote while checked out
    s.set_label(TID, "staging", "Ann", "IV")
    s.close()
    with pytest.raises(SystemExit, match="changed after checkout"):
        hosting.cmd_checkin(cfg, r)
    assert (cfg.workspace / hosting.CHECKOUT).exists()
    hosting.cmd_checkin(cfg, r, discard=True)
    assert r.active()
    assert web_labels(cfg.workspace / "labels.sqlite") == [("Ann", "1", "IV")]


def test_local_page_refuses_a_hosted_workspace(site):
    cfg, r, _ = site
    with pytest.raises(SystemExit, match="hosted"):
        hosting.guard_local_server(cfg.workspace)
    hosting.cmd_checkout(cfg, r)
    hosting.guard_local_server(cfg.workspace)  # allowed while checked out


def test_old_server_backups_pruned_only_when_copied(site):
    cfg, r, remote = site
    old = remote / "backups" / "labels_2026-01-01_000000_hourly.sqlite"
    old.write_bytes(b"x" * 10)
    other = remote / "backups" / "labels_2026-01-02_000000_hourly.csv"
    other.write_bytes(b"y")
    for f in (old, other):
        os.utime(f, (0, 0))  # 1970: far older than keep_days
    (cfg.workspace / "backups").mkdir(exist_ok=True)
    hosting.prune_remote_backups(cfg, r)  # nothing copied yet
    assert old.exists() and other.exists()
    hosting.cmd_pull(cfg, r)  # copies, then prunes
    assert not old.exists() and not other.exists()
    assert (cfg.workspace / "backups" / old.name).exists()


def open_sessions(db):
    s = Store(db)
    rows = s.db.execute("SELECT annotator FROM sessions WHERE ended_at IS NULL")
    out = [row[0] for row in rows]
    s.close()
    return out


def test_deploy_queues_and_the_timer_applies_it(site):
    cfg, r, remote = site
    server = remote.parent
    s = Store(remote / "labels.sqlite")
    s.start_session("Ann", "staging")  # a tab left open overnight
    s.close()
    hosting.cmd_deploy(cfg, r)
    assert (server / "next" / "QUEUED.json").exists()
    assert (server / "bin" / "apply-update.py").exists()
    assert not (server / "app" / "DEPLOYED.json").exists()  # not live yet
    hosting.apply_queued(cfg, r)  # what the 22:00 timer runs
    live = json.loads((server / "app" / "DEPLOYED.json").read_text())
    assert live["applied"] and not (server / "next").exists()
    assert r.active()
    assert open_sessions(remote / "labels.sqlite") == []  # ended, with an export
    assert list((remote / "exports").glob("*_before-update.csv"))
    hosting.apply_queued(cfg, r)  # nothing queued: does nothing


def test_deploy_now_refuses_while_people_are_online(site, monkeypatch):
    cfg, r, remote = site
    ann = [{"annotator": "Ann", "mode": "staging", "seconds_ago": 4}]
    monkeypatch.setattr(hosting, "online", lambda cfg, r: ann)
    with pytest.raises(SystemExit, match="stays queued"):
        hosting.cmd_deploy(cfg, r, now=True)
    assert (remote.parent / "next" / "QUEUED.json").exists()  # still goes tonight
    hosting.cmd_deploy(cfg, r, now=True, force=True)
    assert (remote.parent / "app" / "DEPLOYED.json").exists()


def test_deploy_cancel_drops_the_queue(site):
    cfg, r, remote = site
    hosting.cmd_deploy(cfg, r)
    hosting.cmd_deploy(cfg, r, cancel=True)
    assert not (remote.parent / "next").exists()
    hosting.apply_queued(cfg, r)
    assert not (remote.parent / "app" / "DEPLOYED.json").exists()


def test_update_while_checked_out_only_swaps_code(site):
    cfg, r, remote = site
    s = Store(remote / "labels.sqlite")
    s.start_session("Ann", "staging")
    s.close()
    hosting.cmd_checkout(cfg, r)
    hosting.cmd_deploy(cfg, r, now=True)
    assert (remote.parent / "app" / "DEPLOYED.json").exists()
    assert not r.active()  # checkin starts it
    assert open_sessions(remote / "labels.sqlite") == ["Ann"]  # database untouched
    hosting.cmd_checkin(cfg, r)  # so the checkout digest still matches
    assert r.active()


def test_people_online_are_warned_then_logged_out(site):
    """The whole sequence against a real page: countdown, closing, forced end."""
    cfg, r, remote = site
    app = App(remote, backup_seconds=3600)
    httpd = ThreadingHTTPServer(("127.0.0.1", cfg.port), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{cfg.port}"

    def get(path):
        with urllib.request.urlopen(base + path) as resp:
            return json.loads(resp.read())

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode())
        req.add_header("Content-Type", "application/json")
        req.add_header("X-Token", app.token)
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    try:
        ann = post("/api/login", {"name": "Ann", "mode": "staging"})["session_id"]
        bob = post("/api/login", {"name": "Bob", "mode": "staging"})["session_id"]
        for sid in (ann, bob):
            assert get(f"/api/ping?session={sid}")["update"] is None
        assert {p["annotator"] for p in get("/api/online")} == {"Ann", "Bob"}
        now = {"now": True, "force": True}
        done = threading.Thread(target=hosting.cmd_deploy, args=(cfg, r), kwargs=now)
        done.start()
        phases = []
        deadline = time.time() + 30
        while time.time() < deadline:
            u = get(f"/api/ping?session={ann}")["update"]
            if u and u["phase"] not in phases:
                phases.append(u["phase"])
            if u and u["phase"] == "closing":
                post("/api/logout", {"session_id": ann})  # Ann's page logs out
                break
            time.sleep(0.2)
        assert phases == ["warning", "closing"]
        done.join(30)  # Bob's page never answers: ended when the page stops
        assert not done.is_alive()
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.store.close()
    assert open_sessions(remote / "labels.sqlite") == []
    assert not (remote / "UPDATE_NOTICE.json").exists()
    assert (remote.parent / "app" / "DEPLOYED.json").exists() and r.active()


def test_update_from_old_live_code_without_end_sessions(site):
    """The first update replaces code that has no --end-sessions: still works."""
    cfg, r, remote = site
    old = remote.parent / "app/src/eyespy_spermatogenesis/staging/annotate/server.py"
    old.write_text("raise SystemExit('old code: no --end-sessions')\n")
    s = Store(remote / "labels.sqlite")
    s.start_session("Ann", "staging")
    s.close()
    hosting.cmd_deploy(cfg, r)
    hosting.apply_queued(cfg, r)
    assert "SystemExit('old code" not in old.read_text()  # new code is live
    assert open_sessions(remote / "labels.sqlite") == []
