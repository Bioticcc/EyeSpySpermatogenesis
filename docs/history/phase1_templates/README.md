# Examples

Phase 1 provides a metadata inspection command; see
`docs/history/PHASE1_IO_VALIDATION.md`. Keep private images, manifests and reports outside
tracked source (`data/` and `outputs/` are Git-ignored).

`pilot_manifest.csv` and `roi_requests.csv` contain headers only. Copy them to a
local data/results folder before filling them in. They do not represent actual
images, selected crops or expert labels, and the inspector does not read them.
Leave unknown biological/acquisition fields blank. ROI bounds use the explicitly
requested image level until a source-coordinate transform has been verified.

The overview prototype is described in `docs/OVERVIEW_PROTOTYPE.md`. Its repeatable
`--roi X0 Y0 X1 Y1` argument uses **native source** coordinates. The earlier CSV
ROI template describes requested-level coordinates and is not consumed by that
CLI; convert and verify coordinates explicitly. No real images are bundled.
