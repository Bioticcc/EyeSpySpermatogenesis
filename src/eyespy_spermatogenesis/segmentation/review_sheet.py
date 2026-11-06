"""Build a local click-review page for a batch of overview runs.

Usage::

    python -m eyespy_spermatogenesis.segmentation.review_sheet outputs/RUN_PARENT

``RUN_PARENT`` holds one subdirectory per source, each with a ``manifest.json``
written by ``overview_pipeline --detect --thumbnail-size N``. The page
(``review.html``) shows every candidate thumbnail. The reviewer gives each one a
verdict and exports a CSV. Verdicts are kept in the browser's local storage
only until exported. The page reads local files; it is never published, because
it contains lab images.

The review measures the stage-1 goal directly: does each crop show one tubule,
centred and filling most of the frame?
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from importlib import resources
from pathlib import Path

VERDICTS = [
    ("good", "Good", "The yellow outline traces one whole tubule"),
    ("merged", "Merged", "The yellow outline holds two or more tubules"),
    ("split", "Split", "The outline covers only part of a tubule"),
    ("partial", "Partial / edge", "Tubule cut by the section edge or damaged"),
    ("not_tubule", "Not a tubule", "Interstitium, capsule, vessel, debris"),
    ("unsure", "Unsure", "Cannot decide from the crop"),
    (
        "spill",
        "Outline spills",
        "One tubule, but the outline leaks into background and spoils the crop",
    ),
]


def collect(parent: Path) -> list[dict]:
    sources = []
    for manifest_path in sorted(parent.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete_needs_review":
            print(f"Skipping {manifest_path.parent.name}: {manifest.get('status')}")
            continue
        folder = manifest_path.parent.name
        candidates = []
        for c in manifest["candidates"]:
            candidates.append(
                {
                    "label": c["label"],
                    "instance_id": c.get("instance_id"),
                    "thumbnail": (
                        f"{folder}/{c['thumbnail']}" if c.get("thumbnail") else None
                    ),
                    # v0.3 runs before 2026-11-06 burned the outline into the JPEG.
                    "outline": (
                        f"{folder}/{c['thumbnail_outline']}"
                        if str(c.get("thumbnail_outline", "")).endswith(".png")
                        else None
                    ),
                    "recommended": bool(c.get("crop_recommended")),
                    "flags": c.get("flags", []),
                    "export_status": c.get("export_status"),
                    "source_bounds": c.get("source_bounds"),
                }
            )
        candidates.sort(key=lambda c: (not c["recommended"], c["label"]))
        sources.append(
            {
                "folder": folder,
                "name": Path(manifest["source"]["path"]).name,
                "sha256": manifest["source"].get("sha256"),
                "method": manifest.get("segmentation", {}).get("method"),
                "overlay": f"{folder}/review_overlay.png",
                "candidates": candidates,
            }
        )
    return sources


def write_template(parent: Path, sources: list[dict]) -> Path:
    path = parent / "review_template.csv"
    with open(path, "x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source",
                "source_sha256",
                "label",
                "instance_id",
                "crop_recommended",
                "flags",
                "verdict",
            ]
        )
        for s in sources:
            for c in s["candidates"]:
                writer.writerow(
                    [
                        s["name"],
                        s["sha256"],
                        c["label"],
                        c["instance_id"],
                        c["recommended"],
                        ";".join(c["flags"]),
                        "",
                    ]
                )
    return path


def render(sources: list[dict], title: str) -> str:
    data = json.dumps({"title": title, "sources": sources, "verdicts": VERDICTS})
    data = data.replace("</", "<\\/")
    page = resources.files(__package__).joinpath("review_sheet.html").read_text("utf-8")
    return (
        page.replace("__TITLE__", html.escape(title))
        .replace("__DATA__", data)
        .replace(
            "__LEGEND__",
            "".join(
                f"<li><kbd>{i}</kbd> <b>{html.escape(label)}</b> "
                f"<span>{html.escape(help_)}</span></li>"
                for i, (_, label, help_) in enumerate(VERDICTS, 1)
            ),
        )
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("parent", help="Directory containing per-source run folders")
    args = parser.parse_args(argv)
    parent = Path(args.parent)
    sources = collect(parent)
    if not sources:
        print("No completed runs found", file=sys.stderr)
        return 2
    page = parent / "review.html"
    try:
        with open(page, "x") as handle:
            handle.write(render(sources, parent.name))
        template = write_template(parent, sources)
    except FileExistsError as exc:
        print(f"Refusing to overwrite {exc.filename}", file=sys.stderr)
        return 1
    total = sum(len(s["candidates"]) for s in sources)
    print(f"Wrote {page} and {template} ({len(sources)} sources, {total} crops)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
