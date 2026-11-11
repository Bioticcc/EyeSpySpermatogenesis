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

**Overview** (model review only) shows a chosen section whole: the paper stages
and every model of every tested method side by side, stage colours or
agreement with the paper, plus per-section and overall statistics. It names
the models, so open it after blind review, not before.

Adding a review run (after `round1.evaluate`; restart the server afterwards):

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
