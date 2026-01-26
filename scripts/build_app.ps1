$ErrorActionPreference = "Stop"

# Ensure dependencies are installed via Poetry in the current shell before running.

$Spec = "tubule_app.spec"
$SegPath = "models/exported/seg_model.pt"
if (-not (Test-Path $SegPath)) {
    Write-Warning "Segmentation weights not found at $SegPath. Build will proceed, but predictions will be disabled until weights are added next to the exe."
}

poetry run pyinstaller --noconfirm $Spec
Write-Host "Build complete. Bundle is in dist/TubuleDetectorApp/ (one-folder). Zip that directory for users."
