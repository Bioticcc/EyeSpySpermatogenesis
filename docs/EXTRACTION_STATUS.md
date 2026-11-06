# Tubule extraction: status and handoff (2026-11-06)

This is the starting point for anyone picking up from tubule extraction,
including the staging work that comes next.

## What exists

Running `overview_pipeline --detect` on a whole-section TIFF gives one candidate
per tubule. Each candidate has:

- an outline on an 8× area-averaged overview;
- a square crop of original native pixels, centred on the tubule (`candidate_NNNNN.tif`);
- the overview outline projected onto that crop as a mask (`seed_NNNNN.tif`);
- a `manifest.json` entry with source coordinates, quality measures, flags and
  provenance.

The method is `membrane_ridge_watershed_v0.5` (`segmentation/membrane.py`). It
outlines each tubule along its PAS-positive basement membrane, cleans up the
regions, and re-checks doubtful regions against native pixels. Method history
and every setting are in [OVERVIEW_PROTOTYPE.md](OVERVIEW_PROTOTYPE.md).

Unflagged candidates are `crop_recommended`. These are the crops intended for
analysis. Flagged candidates are kept, but they need a look first.

## How good is it

| Review | Sections (not used for tuning) | Recommended crops judged good |
|---|---|---|
| v0.3 | 07–09 | 93% (336/360) |
| v0.4 | 10–11 | 89% (258/290) |

- v0.5 fixes part of the v0.4 errors. It was checked by replaying the earlier
  verdicts; nobody has reviewed it fresh.
- Sections 04, 05, 06, 12 and 13 have never been reviewed.
- Missed tubules (recall) are unmeasured.

Full records are in [VALIDATION.md](VALIDATION.md#membrane-network-method-owner-click-reviews--2026-11-06).

**Known errors to expect in the crops:**

- **One tubule cut in two.** This is the most common remaining error, especially
  in sections with large oblique tubules. The acrosome ring partway through the
  epithelium can look like a wall.
- **Two adjacent tubules merged.** This happens when their shared wall is too
  faint to detect, even at 2× native.
- **Small outline leaks into gaps or debris.** They are usually harmless.
- **Outlines are only approximate.** They come from the overview, so they can be
  off by about one overview pixel (8 native pixels). They are not verified
  boundaries; use them as rough masks only.

## Current outputs (local, Git-ignored)

| Folder | Contents |
|---|---|
| `outputs/tubule-review-2026-11-06-05/` | **v0.5 on all 13 sections:** 3,734 candidates, 3,245 crop-recommended, 142 fine-scale splits and 13 re-joins; about 21–26 s per section. Thumbnails, outline overlays and an unreviewed `review.html`. No full-resolution crops yet. |
| `outputs/tubule-review-2026-11-06-02/` … `-04/` | v0.2–v0.4 runs, each with the owner's verdict CSV |
| `outputs/tubule-review-2026-11-06-01/`, `outputs/tubule-investigation-2026-11-06-01/` | The failed v0.1 run and its diagnosis |

The 13 source TIFFs are in `Inputs/Staging tubules/`. Never modify them.

## Producing crops for staging

Review runs used `--skip-crops`. To write the full-resolution crops and masks
for a section, run without it, into a new folder:

```bash
.venv/bin/python -m eyespy_spermatogenesis.overview_pipeline \
  "Inputs/Staging tubules/SECTION.tif" --series 0 --downsample 8 --detect \
  --output outputs/crops-YYYY-MM-DD-NN/source_NN
```

- Each section takes roughly 25–30 s, plus crop writing.
- Crops are uncompressed RGB TIFFs: median about 2 MB each (1.3–4.3 MB for most).
  That is roughly 0.7–0.8 GB per section, or about 10 GB for all 13. Write only
  the sections you need.
- Output folders are never overwritten.
- Use `crop_recommended` to choose crops. Keep `instance_id` and `source_bounds`,
  so stage labels can be traced back to their position in the section.

## Open questions for the staging phase

- **Matching existing annotations.** These may be legacy polygons, staged
  images, or something else. To match them to these crops, both must use the
  same coordinates: same source file, same series, same source pixels.
  Annotations made on separately exported 40× images need a registration step
  first; nothing here does that yet.
- **Raw or masked crops.** Should staging see the raw square crop, or the masked
  tubule? Compare both (see [STAGING_STRATEGY.md](STAGING_STRATEGY.md)).
- **Physical scale and counterstain.** These are still unconfirmed
  ([EXPERIMENTAL_SPECIFICATIONS.md](EXPERIMENTAL_SPECIFICATIONS.md)). Any
  pretrained staging model's scale assumptions need checking against them.
