"""Annotation store and server, on a synthetic two-tubule workspace."""

import csv
import io
import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from eyespy_spermatogenesis.staging.annotate.server import App, make_handler
from eyespy_spermatogenesis.staging.annotate.store import PAPER, Store


def tubule(i, recommended=1):
    return {
        "tubule_id": f"src/series-0/candidate-{i}",
        "seq": i,
        "source_folder": "source_01",
        "source_name": "a.tif",
        "source_sha256": "x",
        "label": i,
        "recommended": recommended,
        "flags": "",
        "source_bounds": json.dumps([0, 0, 10, 10]),
        "image": f"images/source_01/c_{i:05d}.jpg",
        "outline": f"images/source_01/o_{i:05d}.png",
        "display_scale": 1.0,
    }


@pytest.fixture
def ws(tmp_path):
    s = Store(tmp_path / "labels.sqlite")
    s.add_tubules([tubule(1), tubule(2, recommended=0)])
    s.close()
    (tmp_path / "images" / "source_01").mkdir(parents=True)
    (tmp_path / "images" / "source_01" / "c_00001.jpg").write_bytes(b"jpg")
    return tmp_path


def test_store_labels_events_and_csv(ws):
    s = Store(ws / "labels.sqlite")
    tid = "src/series-0/candidate-1"
    s.set_label(tid, "staging", "Ann", "XI")
    s.set_label(tid, "staging", "Ann", "XII")  # change keeps one current row
    s.set_label(tid, "staging", "Bo", "XII")
    with pytest.raises(ValueError):
        s.set_label(tid, "staging", "Ann", "XIII")
    with pytest.raises(ValueError):
        s.set_label(tid, "staging", " ", "I")
    with pytest.raises(KeyError):
        s.set_label("missing", "staging", "Ann", "I")
    labs = s.labels("staging")
    assert sorted((r["annotator"], r["value"]) for r in labs) == [
        ("Ann", "XII"),
        ("Bo", "XII"),
    ]
    events = s.db.execute("SELECT value, previous FROM events WHERE action='set'")
    assert [tuple(r) for r in events] == [("XI", None), ("XII", "XI"), ("XII", None)]
    s.clear_label(tid, "staging", "Bo")
    rows = list(csv.DictReader(io.StringIO(s.csv_text())))
    assert [(r["annotator"], r["value"], r["candidate_label"]) for r in rows] == [
        ("Ann", "XII", "1")
    ]
    backup = s.backup(ws / "backups", "test")
    assert backup.exists() and backup.with_suffix(".csv").exists()
    copy = Store(backup)
    assert len(copy.labels()) == 1
    s.close()


def test_paper_label_kept_separate_from_web(ws):
    s = Store(ws / "labels.sqlite")
    tid = "src/series-0/candidate-1"
    s.set_label(tid, "staging", "Soo", "XI", origin=PAPER, detail={"needs_check": True})
    s.set_label(tid, "staging", "Soo", "XII")
    vals = {(r["origin"], r["value"]) for r in s.labels("staging")}
    assert vals == {("paper", "XI"), ("web", "XII")}
    s.close()


def test_paper_import_records_transcriber_and_merges_marks(ws, tmp_path):
    trial = tmp_path / "trial"
    (trial / "snippets").mkdir(parents=True)
    (trial / "summary.json").write_text(json.dumps({"scan": "sheet.jpg"}))
    cols = ["mark", "snippet", "kind", "stage_read", "read_flags", "instance_id"]
    marks = [
        [1, "snippets/m1.jpg", "circled", "VII", "", "src/series-0/candidate-1"],
        [2, "snippets/m2.jpg", "circled", "VII", "", "src/series-0/candidate-1"],
        [3, "snippets/m3.jpg", "circled", "IV", "", "src/series-0/candidate-2"],
        [4, "snippets/m4.jpg", "circled", "V", "", "src/series-0/candidate-2"],
    ]
    with open(trial / "marks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([*cols, "transcribed_by"])
        for m in marks:
            w.writerow([*m, "Reader X"])
            (trial / m[1]).write_bytes(b"jpg")
    from eyespy_spermatogenesis.staging.annotate.paper_import import import_paper

    assert import_paper(ws, trial, "Yeongseok")["imported"] == 2
    s = Store(ws / "labels.sqlite")
    got = {r["tubule_id"][-1]: r for r in s.labels("staging")}
    s.close()
    assert got["1"]["value"] == "VII"
    assert got["1"]["detail"]["transcribed_by"] == "Reader X"
    assert "several_marks_same_stage" in got["1"]["detail"]["read_flags"]
    # marks in one candidate that disagree: no stage, flagged for a person
    assert got["2"]["value"] is None and got["2"]["detail"]["needs_check"]


@pytest.fixture
def server(ws):
    app = App(ws, backup_seconds=3600)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield app, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    app.store.close()


def call(base, path, body=None, token=None):
    req = urllib.request.Request(base + path)
    if body is not None:
        req.data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("X-Token", token)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_session_label_logout(server, ws):
    app, base = server
    code, html = call(base, "/app")
    assert code == 200 and app.token.encode() in html
    assert call(base, "/api/login", {"name": "Ann", "mode": "staging"})[0] == 403
    code, body = call(base, "/api/login", {"name": "Ann", "mode": "staging"}, app.token)
    sid = json.loads(body)["session_id"]
    tid = "src/series-0/candidate-1"
    code, _ = call(
        base,
        "/api/label",
        {"session_id": sid, "tubule_id": tid, "value": "I"},
        app.token,
    )
    assert code == 200
    code, _ = call(
        base,
        "/api/label",
        {"session_id": sid, "tubule_id": tid, "value": "bad"},
        app.token,
    )
    assert code == 400
    labs = json.loads(call(base, f"/api/labels?mode=staging&session={sid}")[1])
    assert [(r["mine"], r["value"]) for r in labs] == [(True, "I")]
    code, body = call(base, "/api/logout", {"session_id": sid}, app.token)
    out = json.loads(body)
    assert out["labels_this_session"] == 1
    assert "Ann" in open(out["export"]).read()
    code, _ = call(
        base,
        "/api/label",
        {"session_id": sid, "tubule_id": tid, "value": "I"},
        app.token,
    )
    assert code == 401


def test_token_survives_restart_so_open_tabs_keep_saving(server, ws):
    """A deploy or checkin restarts the page; an open tab's token must still work."""
    app, base = server
    code, sid = call(base, "/api/login", {"name": "Ann", "mode": "staging"}, app.token)
    assert code == 200
    sid = json.loads(sid)["session_id"]
    again = App(ws, backup_seconds=3600)  # the restarted page
    assert again.token == app.token
    assert (ws / "token.txt").stat().st_mode & 0o777 == 0o600
    again.store.close()
    code, body = call(base, "/api/token")
    assert code == 200 and json.loads(body)["token"] == app.token


def test_ping_reports_build_notice_and_who_is_online(server, ws):
    app, base = server
    code, body = call(base, "/api/login", {"name": "Ann", "mode": "staging"}, app.token)
    sid = json.loads(body)["session_id"]
    code, html = call(base, "/app")
    assert app.build.encode() in html  # the page knows which code it runs
    ping = json.loads(call(base, f"/api/ping?session={sid}")[1])
    assert ping == {"build": app.build, "update": None}
    online = json.loads(call(base, "/api/online")[1])
    assert [o["annotator"] for o in online] == ["Ann"]
    (ws / "UPDATE_NOTICE.json").write_text(
        json.dumps({"phase": "warning", "deadline": time.time() + 120})
    )
    update = json.loads(call(base, "/api/ping")[1])["update"]
    assert update["phase"] == "warning" and 115 <= update["seconds_left"] <= 120
    call(base, "/api/logout", {"session_id": sid}, app.token)
    assert json.loads(call(base, "/api/online")[1]) == []  # logged out = gone


def test_guest_is_not_an_annotator_name(server):
    """Guest labels are never saved, so no labels may be filed under "Guest"."""
    app, base = server
    for name in ("Guest", " guest ", "GUEST"):
        login = {"name": name, "mode": "staging"}
        code, body = call(base, "/api/login", login, app.token)
        assert code == 400 and b"own name" in body


def test_labels_are_anonymous_on_the_page(server, ws):
    """Decision 030: the page never learns who gave which label; only counts."""
    app, base = server
    tid1, tid2 = "src/series-0/candidate-1", "src/series-0/candidate-2"
    sids = {}
    for name, value in (("Ann", "IV"), ("Bob", "V")):
        _, body = call(base, "/api/login", {"name": name, "mode": "staging"}, app.token)
        sids[name] = json.loads(body)["session_id"]
        for tid in (tid1, tid2):
            lab = {"session_id": sids[name], "tubule_id": tid, "value": value}
            assert call(base, "/api/label", lab, app.token)[0] == 200
    app.store.set_label(
        tid1, "staging", "Yeongseok", "VI", origin="paper",
        detail={"transcription_checked_by": "Ann", "needs_check": False},
    )  # fmt: skip
    raw = call(base, f"/api/labels?mode=staging&session={sids['Ann']}")[1]
    for name in (b"Ann", b"Bob", b"Yeongseok", b"annotator", b"updated_at"):
        assert name not in raw
    labs = json.loads(raw)
    assert sorted((r["value"], r["mine"]) for r in labs if r["origin"] == "web") == [
        ("IV", True), ("IV", True), ("V", False), ("V", False),
    ]  # fmt: skip
    paper = [r for r in labs if r["origin"] == "paper"][0]
    assert paper["detail"]["transcription_checked"] is True
    # without a session nothing is "mine"
    anon = json.loads(call(base, "/api/labels?mode=staging")[1])
    assert not any(r["mine"] for r in anon)
    # the only per-person figures: counts, in total and per stage
    people = json.loads(call(base, "/api/progress?mode=staging")[1])
    got = {(p["name"], p["origin"]): (p["n"], p["values"]) for p in people}
    assert got[("Ann", "web")] == (1, {"IV": 1})  # candidate-2 is flagged: not counted
    assert got[("Yeongseok", "paper")] == (1, {"VI": 1})
    # no CSV download on the page; review files lose their names
    assert call(base, "/api/export.csv")[0] == 404
    (ws / "review" / "eyespy").mkdir(parents=True)
    index = {
        "sources": [{"annotator": "Yeongseok", "origin": "paper", "n": 3}],
        "note": "Yeongseok's paper labels; checked by Ann",
    }
    (ws / "review" / "eyespy" / "index.json").write_text(json.dumps(index))
    served = json.loads(call(base, "/ws/review/eyespy/index.json")[1])
    assert served == {
        "sources": [{"origin": "paper", "n": 3}],
        "note": "an annotator's paper labels; checked by an annotator",
    }


def test_online_list_is_not_served_through_the_tunnel(server):
    app, base = server
    req = urllib.request.Request(base + "/api/online")
    req.add_header("Cf-Connecting-Ip", "203.0.113.5")
    try:
        urllib.request.urlopen(req)
        raise AssertionError("served through the tunnel")
    except urllib.error.HTTPError as e:
        assert e.code == 404
    assert call(base, "/api/online")[0] == 200  # from the server itself


def test_short_or_missing_token_file_is_replaced(ws):
    (ws / "token.txt").write_text("short\n")
    app = App(ws, backup_seconds=3600)
    assert len(app.token) >= 32 and (ws / "token.txt").read_text().strip() == app.token
    app.store.close()


def test_landing_page_is_the_guide(server, ws):
    app, base = server
    code, html = call(base, "/")
    assert code == 200 and b"How to stage tubules" in html
    assert app.token.encode() not in html
    assert b'href="/app"' in html
    (ws / "guide").mkdir()
    (ws / "guide" / "01-login.png").write_bytes(b"png")
    assert call(base, "/ws/guide/01-login.png") == (200, b"png")


def test_server_only_serves_workspace_images(server):
    _, base = server
    assert call(base, "/ws/images/source_01/c_00001.jpg") == (200, b"jpg")
    assert call(base, "/ws/images/../labels.sqlite")[0] == 404
    assert call(base, "/ws/labels.sqlite")[0] == 404
    assert call(base, "/static/../server.py")[0] == 404


def test_model_review_answers_and_csv(server, ws):
    app, base = server
    tid = "src/series-0/candidate-1"
    app.store.set_label(tid, "staging", "Yeongseok", "VII", origin=PAPER)
    app.store.add_predictions(
        "run-1",
        [
            {"tubule_id": tid, "ord": 0, "slot": "A", "model": "m1", "value": "VII"},
            {"tubule_id": tid, "ord": 0, "slot": "B", "model": "m2", "value": "VIII"},
        ],
    )
    assert json.loads(call(base, "/api/config")[1])["review_runs"] == ["run-1"]
    preds = json.loads(call(base, "/api/predictions?run=run-1")[1])
    assert {(p["slot"], p["value"]) for p in preds} == {("A", "VII"), ("B", "VIII")}
    _, body = call(
        base, "/api/login", {"name": "Ann", "mode": "model_review"}, app.token
    )
    sid = json.loads(body)["session_id"]
    lab = {"session_id": sid, "tubule_id": tid, "value": "VIII"}
    assert call(base, "/api/label", {**lab, "run": "nope"}, app.token)[0] == 400
    assert call(base, "/api/label", {**lab, "run": "run-1"}, app.token)[0] == 200
    # review answers are their own mode, not staging labels
    staging = json.loads(call(base, "/api/labels?mode=staging")[1])
    assert [r["origin"] for r in staging] == ["paper"]
    rows = list(csv.DictReader(io.StringIO(app.store.csv_text())))
    rv = [r for r in rows if r["mode"] == "model_review"][0]
    assert (rv["value"], rv["review_run"], rv["paper_stage"]) == (
        "VIII",
        "run-1",
        "VII",
    )
    assert (rv["model_a"], rv["model_a_stage"], rv["model_b_stage"]) == (
        "m1",
        "VII",
        "VIII",
    )
    # re-adding a run replaces its predictions but keeps the answers
    app.store.add_predictions("run-1", [])
    assert app.store.predictions("run-1") == []
    assert len(app.store.labels("model_review")) == 1


def test_server_serves_review_overview_files(server, ws):
    app, base = server
    d = ws / "review" / "run-1"
    d.mkdir(parents=True)
    (d / "overview.json").write_text('{"run": "run-1"}')
    code, body = call(base, "/ws/review/run-1/overview.json")
    assert code == 200 and json.loads(body)["run"] == "run-1"
    assert call(base, "/ws/review/../labels.sqlite")[0] == 404


def test_config_lists_overview_runs(server, ws):
    app, base = server
    for run in ("staging-round2-x", "staging-round1-x"):
        d = ws / "review" / run
        d.mkdir(parents=True)
        (d / "overview.json").write_text("{}")
    (ws / "review" / "no-overview").mkdir()
    cfg = json.loads(call(base, "/api/config")[1])
    assert cfg["overview_runs"] == ["staging-round1-x", "staging-round2-x"]
