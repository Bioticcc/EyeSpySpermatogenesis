# PyInstaller spec for Tubule Stage GUI (one-folder)
# Bundles Python, dependencies, and exported model weights.

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

project_root = Path(__file__).parent.resolve()
entry_script = project_root / "src" / "app" / "gui.py"

datas = []

# Bundle model weights if present.
model_path = project_root / "models" / "exported" / "model.pt"
if model_path.exists():
    datas.append((str(model_path), "models/exported"))
else:
    print(f"WARNING: Model weights not found at {model_path}. Build will proceed without weights.", file=sys.stderr)

# Include tifffile data and torch/torchvision assets to keep runtime dependencies intact.
datas += collect_data_files("tifffile")
datas += collect_data_files("torch")
datas += collect_data_files("torchvision")

hiddenimports = collect_submodules("torch") + collect_submodules("torchvision")
binaries = collect_dynamic_libs("torch") + collect_dynamic_libs("torchvision")

a = Analysis(
    scripts=[str(entry_script)],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TubuleStageApp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="TubuleStageApp",
)
