"""Import paper stages transferred by the alignment trial into a workspace.

Usage::

    python -m eyespy_spermatogenesis.staging.annotate.paper_import \
        WORKSPACE outputs/paper-alignment-2026-11-06-01 --annotator Yeongseok

Reads ``marks.csv`` (and ``snippets/``) written by ``paper_align.py`` (or a
later reader; an optional ``transcribed_by`` column names who read the
numerals, otherwise the v1 machine reader is assumed). Each
candidate with marks gets one ``origin='paper'`` staging label for the
annotator. Reads are machine transcriptions of handwriting: any flagged read,
a candidate with several marks, or an unreadable mark is stored with
``needs_check``. If marks in one candidate disagree, no stage is stored.
Re-importing never overwrites a transcription a person has already checked.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from eyespy_spermatogenesis.staging.annotate.store import PAPER, Store

V1_READER = "paper_align.py (machine read)"


def import_paper(workspace: Path, trial_dir: Path, annotator: str) -> dict:
    store = Store(workspace / "labels.sqlite")
    known = {t["tubule_id"] for t in store.tubules()}
    summary = json.loads((trial_dir / "summary.json").read_text())
    scan_name = Path(summary["scan"]).stem
    snip_dest = workspace / "paper" / scan_name
    snip_dest.mkdir(parents=True, exist_ok=True)
    existing = {
        lab["tubule_id"]: lab
        for lab in store.labels("staging")
        if lab["origin"] == PAPER and lab["annotator"] == annotator
    }
    groups = defaultdict(list)
    with open(trial_dir / "marks.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["instance_id"]:
                groups[row["instance_id"]].append(row)
    stats = defaultdict(int)
    for tid, marks in groups.items():
        if tid not in known:
            stats["candidate_not_in_workspace"] += 1
            continue
        if existing.get(tid, {}).get("detail", {}).get("transcription_checked_by"):
            stats["kept_checked_transcription"] += 1
            continue
        snippets = []
        for m in marks:
            src = trial_dir / m["snippet"]
            name = Path(m["snippet"]).name
            shutil.copy2(src, snip_dest / name)
            snippets.append(f"paper/{scan_name}/{name}")
        flags = sorted({f for m in marks for f in m["read_flags"].split(";") if f})
        reads = {m["stage_read"] for m in marks}
        value = None
        if len(marks) > 1:
            if len(reads) == 1 and "" not in reads:
                flags.append("several_marks_same_stage")
                value = reads.pop()
            else:
                flags.append("marks_disagree_or_unreadable")
        else:
            value = marks[0]["stage_read"] or None
        detail = {
            "scan": summary["scan"],
            "marks": [int(m["mark"]) for m in marks],
            "mark_kinds": [m["kind"] for m in marks],
            "stage_reads": [m["stage_read"] for m in marks],
            "read_flags": flags,
            "snippets": snippets,
            "needs_check": bool(flags) or value is None,
            "transcribed_by": marks[0].get("transcribed_by") or V1_READER,
        }
        store.set_label(
            tid,
            "staging",
            annotator,
            value,
            origin=PAPER,
            actor="paper-import",
            detail=detail,
            action="paper_import",
        )
        stats["imported"] += 1
        stats["needs_check"] += detail["needs_check"]
    store.close()
    return dict(stats)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("workspace", type=Path)
    p.add_argument("trial_dir", type=Path)
    p.add_argument("--annotator", required=True)
    a = p.parse_args(argv)
    print(json.dumps(import_paper(a.workspace, a.trial_dir, a.annotator), indent=1))


if __name__ == "__main__":
    main()
