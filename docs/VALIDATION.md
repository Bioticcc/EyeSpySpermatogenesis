# Phase 0 verification

Validation date: **2026-11-04**. This record covers repository/scaffold checks,
not image compatibility or biological performance.

## Environment and completed checks

- Linux, Python 3.11.2; local `.venv`.
- Editable installation: `python -m pip install -e '.[dev]'` succeeded.
- Resolved development tools: pytest 8.4.2, Ruff 0.17.3. These are the observed
  local versions, not a hash-locked deployment environment.
- Ruff lint passed; formatting check passed after formatting the import test.
- Package and all six planned submodules imported: **1 smoke test passed**.
- `pip check` found no broken requirements.
- Runtime dependency list is empty. No image/ML/GUI dependencies were installed.

The GitHub Actions workflow covers Python 3.11–3.13 on Linux. Hosted CI results
must be checked separately after publication; local checks cover only Python 3.11.

## Archive and publication

Archival commit/tree, original-file counts and local-only inventory are in
[GIT_MIGRATION.md](GIT_MIGRATION.md). The remote archive was verified at the exact
original SHA before changing main. Full fetch of original historical objects
completed; `git fsck --full` reported no integrity errors (its unborn-main notice
was expected before the scaffold commit). All **743 reachable legacy objects**
are present, with **zero missing objects**. This is a complete local archive,
not a promise to fetch missing blobs later. The historical object transfer was
substantially larger than GitHub's reported repository size; Git reported about
963 MiB of packs plus temporary/duplicate loose objects before automatic packing.
These objects remain inside `.git`, outside the active project file tree.

Publication and final tree checks are added after the reviewed commit is created.

## Scientific and platform limits

No representative image headers were inspected. No reader was validated on actual
OME-TIFF variants, no crops/masks were generated, no classifiers were run, and no
scientific accuracy or memory-budget result exists. Third-party capabilities in
the software evaluation are source/documentation evidence. No native application,
Windows/macOS/WSL deployment or disconnected installation was tested.

Phase 1 starts with representative local file paths, staining/acquisition records,
scale verification and owner authorization for characterization. Major algorithms
remain gated on an approved prototype plan.
