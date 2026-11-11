#!/usr/bin/env bash
# Segment every TIFF in a folder with the v0.5 settings used for staging, ready
# for the annotation page. Sources are only read. TIFFs whose contents
# (sha256) are already in the annotation workspace are skipped, so a section
# is never added twice.
# Usage: scripts/segment_for_staging.sh "Inputs/Staging set 2" [RUN_NAME] [WORKSPACE]
# Output: outputs/RUN_NAME/source_NN/ (one per new TIFF, sorted by file name).
set -euo pipefail
cd "$(dirname "$0")/.."
IN="${1:?give the folder of TIFFs}"
RUN="${2:-tubule-staging-$(date +%F)-01}"
WS="${3:-outputs/staging-annotation-2026-11-06-01}"
OUT="outputs/$RUN"
if [ -e "$OUT" ]; then echo "$OUT already exists; choose a new RUN_NAME" >&2; exit 1; fi
mapfile -t ALL < <(find -L "$IN" -maxdepth 1 -type f \( -iname '*.tif' -o -iname '*.tiff' \) | sort)
if [ "${#ALL[@]}" -eq 0 ]; then echo "No TIFFs in $IN" >&2; exit 1; fi
# Keep only TIFFs not yet in the workspace (and not repeated in this folder).
mapfile -t TIFFS < <(.venv/bin/python - "$WS" "${ALL[@]}" <<'EOF'
import hashlib, sqlite3, sys
from pathlib import Path
ws, files = Path(sys.argv[1]), sys.argv[2:]
known = {}
if (ws / "labels.sqlite").exists():
    db = sqlite3.connect(f"file:{ws / 'labels.sqlite'}?mode=ro", uri=True)
    known = dict(db.execute("SELECT DISTINCT source_sha256, source_name FROM tubules"))
seen = {}
for f in files:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    sha = h.hexdigest()
    if sha in known:
        print(f"skip (already on the page as {known[sha]}): {Path(f).name}", file=sys.stderr)
    elif sha in seen:
        print(f"skip (same file as {seen[sha]}): {Path(f).name}", file=sys.stderr)
    else:
        seen[sha] = Path(f).name
        print(f)
EOF
)
if [ "${#TIFFS[@]}" -eq 0 ]; then echo "Nothing new to segment."; exit 0; fi
echo "Segmenting ${#TIFFS[@]} new TIFF(s) of ${#ALL[@]}."
mkdir -p "$OUT"
i=0
for f in "${TIFFS[@]}"; do
  i=$((i + 1))
  dest=$(printf "%s/source_%02d" "$OUT" "$i")
  echo "[$i/${#TIFFS[@]}] $(basename "$f") -> $dest"
  .venv/bin/python -m eyespy_spermatogenesis.overview_pipeline "$f" \
    --series 0 --downsample 8 --detect --skip-crops --thumbnail-size 384 \
    --output "$dest" >"$dest.log" 2>&1 || echo "  FAILED, see $dest.log"
done
.venv/bin/python -m eyespy_spermatogenesis.segmentation.review_sheet "$OUT" >/dev/null 2>&1 || true
echo
echo "Done: $OUT"
echo "Check the overlays (source_NN/review_overlay.png), then add to the page with:"
echo "  .venv/bin/python -m eyespy_spermatogenesis.staging.annotate.prepare $OUT $WS --jobs 6"
