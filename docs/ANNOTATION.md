# Staging annotation page

A local page where lab experts stage tubules (or review outlines) one crop at a
time. It runs on this workstation only (decision 021) and collects training
labels for a future staging model.

## Start and stop

```bash
scripts/annotate.sh                # start in the background and open the page
pkill -f staging.annotate.server   # stop (writes a final backup)
```

The server keeps running when the terminal closes. It does not restart after a
reboot; run `scripts/annotate.sh` again. Only this computer can open the page.

## For annotators

1. Type your name and choose **Staging**, **Segmentation review** or **Model review**. There is
   no password; the name only labels your work.
2. Press a key and the page saves it and moves on:

   | Key | Stage | Key | Stage |
   |---|---|---|---|
   | 1 | I | 7 | VII |
   | 2 or 3 | II–III | 8 | VIII |
   | 4 | IV | 9 | IX |
   | 5 | V | 0 | X |
   | 6 | VI | - | XI |
   | B | Can't stage / bad crop | = | XII |

   → or Space skips, ← goes back, Delete clears your label, O hides the
   outline, the mouse wheel zooms.
3. Labels from other people are shown beside the image. Everyone can stage
   every tubule; disagreement is data, not an error.
4. **Log out** when you finish. That saves a CSV of all labels.

**Lists.** "Not staged by me" (default) shows tubules nobody has staged first,
then the rest. Staging shows accepted (unflagged) crops only; segmentation
review shows every candidate.

**Section map (staging).** On **Progress**, click a section in "By section".
The whole section opens, each tubule coloured by the stage it has so far (your
label first, then the newest on-screen label, then paper; or only yours),
with its numeral; uncoloured tubules have no stage yet, dark grey ones are
flagged crops that are not staged. It shows at a glance how complete a section
is. Click a tubule to open it in Annotate and label it; labelling then carries
on through the list as normal (if the tubule is not in your current list, the
list switches to "All accepted tubules").

**Segmentation review** shows the crops as a grid, like the original review
sheet (owner, 2026-11-13). Judge the yellow outline. Click a crop or use ← →,
then press a verdict key (1 Good, 2 Merged, 3 Split, 4 Partial / edge, 5 Not a
tubule, 6 Unsure, 7 Outline spills); the next crop is selected. 0 or Delete
clears, O hides outlines, the slider sets the crop size. Dashed borders are
flagged crops. Double-click a crop (or press Enter) for the full-size view;
Esc returns to the grid.

**Paper stages.** Where a hand-staged printout was transferred, a panel shows
the handwritten mark and the machine reading. If the reading is wrong press
**F** then the stage written on the paper; if right, press **T**. This corrects
the transcription of the paper, which is separate from your own stage. The
"Paper readings to check" list holds the doubtful readings.

## Model review

Judges staging models (decision 023 rounds) against the reviewer's own eye.
Each review run shows a fixed, shuffled sample of tubules (up to 15 per paper
stage). Press the stage you judge correct, with the same keys as Staging. The
**Answers** panel shows the paper stage and Models A and B, which are shuffled
per tubule; **Blind mode** hides them until you answer. Every prediction comes
from a model that never saw the tubule's section. Answers are saved like other
labels (mode `model_review`, not staging labels); the CSV has extra columns for
the review run, paper stage and both models' stages. **Progress** shows each
model's and the paper's agreement with each reviewer.

**Current A/B pair (from 2026-11-13): EyeSpy_v1 vs EyeSpy_v2** (review run
`eyespy-v1-vs-v2`, decision 027). Earlier runs compared round 1's and round 3's
extractors (UNI2-h vs H-optimus-0); they stay in the database. The page always
uses the newest run. Every run holds the same 164 tubules, so answers carry
over. A new EyeSpy pair (no server restart needed; reload the page):

```bash
.venv/bin/python -m eyespy_spermatogenesis.staging.eyespy review \
  --workspace outputs/staging-annotation-2026-11-06-01 \
  --models EyeSpy_v1 EyeSpy_v2
```

Any two names from `Model_Checkpoints/registry.json` work, including
smoothed, class-weighted and experiment models.

**Overview** (model review only) has two tabs. Both name the models, so open
them after blind review, not before.

- **Method overview** shows a chosen section whole: the paper stages and
  every model of every tested method side by side, stage colours or agreement
  with the paper, plus per-section and overall statistics.
- **EyeSpy (selected strategy)** (decision 027) shows the primary strategy
  only. One row per sample section (#112 early, #92 middle, #91 late stages):
  the ground truth, then every saved EyeSpy model, in the order they were
  saved. Below: the model history (scores, training labels, smoothing, class
  weighting, intervals against the parent model and the previous version),
  per-stage recall and each model's settings. Untick "Comparison models" to
  hide the class-weighted ones. The page reads
  `review/eyespy/index.json`, written from `Model_Checkpoints/registry.json`.
  Open it directly with `#view=overview&tab=eyespy` after logging in.

**Clicking a tubule** on any panel of either tab (owner, 2026-11-13) opens a
grid of every tubule of that section with the same stage according to that
panel, wrong ones first. Each card shows the tubule crop with its outline, the
panel's stage, the ground truth and a score: green correct, amber one stage
off, red two or more off. The grid can switch stage, group by the
ground-truth stage instead ("what did the model call the true stage VI
tubules?") and show only correct or only wrong ones. Clicking a card opens
the tubule full size in the same zoomable viewer as Annotate, with every
model's stage for it; ← → step through the grid, O toggles the outline, Esc
goes back. All of this is read-only: nothing is labelled here. The ⤢ next to a
panel's title opens the panel image itself. The click data are
`review/maps/<section>.png` (which tubule is where) and per-tubule stages
(`review/<run>/<section>/stages.json`; the EyeSpy `index.json`). Runs made
before 2026-11-13 were backfilled once, and checked against their saved
per-section scores, with
`python -m eyespy_spermatogenesis.staging.annotate.clickmaps backfill WORKSPACE`;
newer runs write them automatically.

Saving a new EyeSpy version (adds three models: plain, smoothed, class
weighted; never overwrites an earlier version). New on-screen labels are
exported to a dated snapshot and embedded first (about 15 min per 300 crops
on the RTX 4060):

```bash
B=/mnt/20TB_A/Adam/EyeSpySpermatogenesisRebooted
E=$B/staging-eyespy/web-YYYY-MM-DD
export HF_HUB_CACHE=$B/models/hf-cache HF_HUB_OFFLINE=1
.venv/bin/python -m eyespy_spermatogenesis.staging.round1.data \
  outputs/staging-annotation-2026-11-06-01 $E/data --origin web
M=eyespy_spermatogenesis.staging.round1.embed
.venv/bin/python -m $M $E/data $E/embeddings --model h-optimus-0 --views s05_raw
.venv/bin/python -m $M $E/data $E/embeddings --model h-optimus-0 --views s05_raw --tta d4
.venv/bin/python -m $M $E/data $E/embeddings --model dinov2-l --views s1_raw --tta d4
.venv/bin/python -m eyespy_spermatogenesis.staging.eyespy train --version 3 \
  --source $B/staging-round1/data:$B/staging-round1/embeddings \
  --source $E/data:$E/embeddings \
  --workspace outputs/staging-annotation-2026-11-06-01 --note "what changed"
```

Label experiments save one plain model tagged "experiment" (the tab's
Experiments box hides them): `--experiment SLUG` with source options
`:shift=N` (move every label of that source N stages; −1 = one earlier) and
`:weight=W` (its tubules count W times in training), e.g.
`--source $E/data:$E/embeddings:shift=-1:weight=3 --experiment web-shift-1-weight-3`.

The newest snapshot of on-screen labels replaces older ones (pass only one
`web-*` source); a tubule labelled in two sources stops the run. Reload the
page afterwards; no server restart is needed.

Adding a review run for the older round 1–3 methods (after `round1.evaluate`; restart the server afterwards):

```bash
R=/mnt/20TB_A/Adam/EyeSpySpermatogenesisRebooted/staging-round1
.venv/bin/python -m eyespy_spermatogenesis.staging.round1.review \
  $R/data $R/embeddings outputs/staging-round1-2026-11-09-01 outputs/staging-annotation-2026-11-06-01
.venv/bin/python -m eyespy_spermatogenesis.staging.round1.overlay \
  $R/data $R/embeddings outputs/staging-round1-2026-11-09-01 outputs/staging-annotation-2026-11-06-01 \
  --method "Step 1: Yeongseok paper labels; frozen features + classifier"
```

When the results hold more than two extractors, pick the A/B pair with
`--models` (round 3, ⚠ UNI2-h is research only, decision 025):

```bash
.venv/bin/python -m eyespy_spermatogenesis.staging.round1.review \
  $R/data $R/embeddings outputs/staging-round3-2026-11-11-01 outputs/staging-annotation-2026-11-06-01 \
  --models uni2-h h-optimus-0
.venv/bin/python outputs/staging-round3-2026-11-11-01/overview.py outputs/staging-annotation-2026-11-06-01
```

Runs drawn with the same seed from the same labels sample the same tubules,
and answers are stored per tubule, so one pass of answers scores every such
run. The page's Annotate tab uses the newest run.

## Where the data is

Everything is in the workspace folder, e.g.
`outputs/staging-annotation-2026-11-06-01/` (Git-ignored):

| Path | Contents |
|---|---|
| `labels.sqlite` | Current labels (one per person per tubule) and an append-only `events` log of every change |
| `backups/` | Database + CSV copies at start-up, every hour, and at shutdown |
| `exports/` | A CSV per log-out and per "Export CSV" click |
| `images/` | Display JPEGs and outline PNGs (viewing only; not for analysis) |
| `paper/` | Snippets of handwritten marks |
| `review/<run>/` | Overview panels and statistics for each model-review run |
| `review/eyespy/` | EyeSpy tab panels and `index.json` (from `Model_Checkpoints/registry.json`) |
| `review/maps/`, `review/sections/` | Click maps (which tubule is where) and overview images per section, for the Overview click-through and the Progress section map |

Use the newest file in `exports/` or `backups/`. CSV columns: `mode,
source_name, source_sha256, candidate_label, tubule_id, recommended, annotator,
origin (web|paper), value, paper_read_flags, paper_transcription_checked_by,
created_at, updated_at, source_bounds`. `tubule_id` and `source_bounds` trace
each label to its native source pixels.

## Adding sections (new TIFFs)

New TIFFs must be segmented before they can be staged: the page only shows
candidates from `overview_pipeline` runs.

1. Put the TIFFs in their own folder under `Inputs/` (Git-ignored), e.g.
   `Inputs/Staging set 2/`. Never edit them; outputs go to `outputs/`.
2. Segment them with the v0.5 settings (about 30 s per section):

   ```bash
   scripts/segment_for_staging.sh "Inputs/Staging set 2" tubule-staging-2026-11-06-01
   ```

   It assumes series 0, as for the current sections; check a new scanner's
   files with `inspect_tiff` first.
3. Look at each `source_NN/review_overlay.png`. Optionally review outlines in
   the page's Segmentation mode after step 4.
4. Add the run to the existing workspace (labels are kept; restart the server
   afterwards so the page sees the new sections):

   ```bash
   .venv/bin/python -m eyespy_spermatogenesis.staging.annotate.prepare \
     outputs/tubule-staging-2026-11-06-01 outputs/staging-annotation-2026-11-06-01 --jobs 6
   ```

**No section is ever added twice.** Sections are matched by file contents
(sha256), not by name. A folder may mix new TIFFs with ones already on the page:
`segment_for_staging.sh` skips the known ones before segmenting, and `prepare`
skips any run whose TIFF is already in the workspace from another run. The
existing section keeps its tubules and labels. (Re-segmenting a labelled TIFF
would renumber its candidates, so old labels would point at different tubules;
use a new workspace if you ever need that.) Within the page each person has
one current stage per tubule; changing it replaces the old one, which stays in
the change log.

Each run's images go to their own folder, so several runs share one workspace.

Paper stages from the alignment trial are imported with:

```bash
.venv/bin/python -m eyespy_spermatogenesis.staging.annotate.paper_import \
  outputs/staging-annotation-2026-11-06-01 outputs/paper-alignment-2026-11-06-01 \
  --annotator Yeongseok
```

Re-importing never overwrites a reading a person has checked.

## Paper sheets

The transfer trial and its limits are in
[STAGING_STRATEGY.md](STAGING_STRATEGY.md#paper-annotation-transfer-trial-2026-11-06).
Sheets 2–8 (2026-11-09) used the v2 workflow in
`outputs/paper-alignment-2026-11-09-01/` (see its `RUN_NOTES.md`). For a new
sheet:

1. `.venv/bin/python scripts/check_new_staging_inputs.py` lists sheets and
   TIFFs not yet on the page. A new TIFF must be segmented and added first
   (above). A re-photo of an imported sheet goes in that folder's
   `duplicates.json` instead.
2. `paper_align_v2.py PHOTO RUN_SOURCE_DIR OUT_DIR` registers the photo and
   assigns marks. Check `overlay_on_overview.png` and that the mark count is
   close to the sheet's legend total.
3. `transcribe_sheets.py prep PHOTO OUT_DIR` writes upright contact sheets;
   write `transcription.txt` (format in the script), check that the page reads
   upright (VIII, not IIIV; else read it rotated or use `--flip`), then
   `transcribe_sheets.py apply OUT_DIR` and compare the counts with the legend.
4. Import with `paper_import` as above.

The v2 script's own numeral reads are not reliable; the transcription step
replaces them.
