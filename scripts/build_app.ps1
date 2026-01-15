$ErrorActionPreference = "Stop"

# Ensure dependencies are installed via Poetry in the current shell before running.

$Spec = "tubule_app.spec"
$ModelPath = "models/exported/detector.pt"
if (-not (Test-Path $ModelPath)) {
    Write-Warning "Detector weights not found at $ModelPath. Build will proceed, but predictions will be disabled until weights are added next to the exe."
}

poetry run pyinstaller --noconfirm $Spec
Write-Host "Build complete. Bundle is in dist/TubuleDetectorApp/ (one-folder). Zip that directory for users."
