# EyeSpySpermatogenesis

A new foundation for reproducible, preferably offline analysis of **mouse testis
histology: PAS staining, FFPE preparation, and native 40× analysis detail**.

The project has two separate objectives:

1. **Tubule isolation:** locate individual seminiferous tubules and export complete
   crops, masks, quality information, and source coordinates.
2. **Epithelial staging:** assess suitable isolated tubules, preserving uncertainty
   instead of forcing a stage assignment.

## Current status

**Phase 0: repository reset, software investigation, and prototype planning.** This
repository contains documentation, empty package modules, and development checks.
OME-TIFF ingestion, cropping, segmentation, staging, and a GUI are **not
implemented**. No representative new-project images have been characterized.

The previous polygon-supervised U-Net experiments did not establish satisfactory
performance. Their code and history are preserved on `legacy`; they are a possible
benchmark, not the new architecture. New `main` replaces the active file tree
while retaining Git ancestry, allowing publication without a history rewrite.
The reset is published on GitHub `main`, with the original project preserved on
`legacy`. Preservation and publication checks are in
[the migration record](docs/GIT_MIGRATION.md).

## Planned inputs and workflow

- **A complete tissue section in a large OME-TIFF.** Read bounded regions without
  materializing the highest-resolution section. Compare overlapping native-detail
  tiles with lower-resolution localization followed by native-detail retrieval.
- **A folder of existing 40× microscopy images.** Validate actual formats and
  metadata; images may contain one or several tubules.

Both pathways will supply the same analysis contract to tubule isolation. Only
suitable crops then enter staging. Nominal **40× does not specify µm/pixel**;
counterstain, calibration, pyramid structure, and stage-group requirements remain
pending. Enlarging a lower-resolution image does not restore native detail.

## Approach

Reuse mature software first. Compare classical morphology/watershed with
QuPath, ilastik, or Fiji integration before considering new deep-learning work.
Evaluate existing bright-field staging systems separately. None is yet validated
on this project's images. No candidate image-processing library is installed by
the scaffold.

Start with [software evidence](docs/SOFTWARE_EVALUATION.md), the
[competing prototypes](docs/PROTOTYPES.md), and the
[evaluation plan](docs/EVALUATION_PLAN.md). Prototype selection needs representative
images and expert review.

## Development

Python 3.11+; there are no runtime dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

On Windows, use `py -3 -m venv .venv` and
`.venv\Scripts\Activate.ps1`, then the same `python -m ...` commands.
`bash scripts/check.sh` runs the three checks above in an activated environment.
An import test checks packaging only; it says nothing about biological accuracy.
Installation requires package access; future offline operation will require
separately prepared, versioned dependencies and any approved models.

## Roadmap and documentation

| Phase | Deliverable |
|---|---|
| 0 — current | Preserve history; scaffold; software review; comparison design |
| 1 | Characterize real images; verify scale; compare bounded region readers |
| 2 | Compare segmentation and whole-section prototypes |
| 3 | Select and validate reproducible tubule extraction |
| 4 | Establish staging feasibility, accuracy, and uncertainty |
| 5 | Integrate both inputs, batch exports, then assess GUI/deployment |

- [Project plan and Phase 1 entry checklist](docs/PROJECT_PLAN.md)
- [Confirmed and pending experimental specifications](docs/EXPERIMENTAL_SPECIFICATIONS.md)
- [Architecture and proposed data contracts](docs/ARCHITECTURE.md)
- [Staging strategy](docs/STAGING_STRATEGY.md)
- [Legacy findings](docs/LEGACY_FINDINGS.md)
- [Decision log](docs/DECISIONS.md)
- [Agent guidance](AGENTS.md)

Related repositories:
[EyeSpySpermatogenesis](https://github.com/Bioticcc/EyeSpySpermatogenesis),
[SCP-SpermColorPixels](https://github.com/Bioticcc/SCP-SpermColorPixels), and
[EyeSpySC](https://github.com/Bioticcc/EyeSpySC).

Project licensing is pending owner selection. No third-party code or model weights
have been incorporated into the new package.
