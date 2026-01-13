$ErrorActionPreference = "Stop"
$ModelPath = "models/exported/model.pt"
if (-not (Test-Path $ModelPath)) {
    Write-Warning "Model weights not found at $ModelPath. The app will still build but predictions will be disabled."
}
poetry run pyinstaller --noconfirm --onedir --name TubuleStageApp --add-data "$ModelPath;models/exported" src/app/gui.py
Write-Host "Build complete. Check dist/TubuleStageApp for the bundled app."
