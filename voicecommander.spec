import os

from PyInstaller.utils.hooks import collect_data_files

nvidia_build = os.environ.get("VOICECOMMANDER_BUILD_NVIDIA") == "1"

a = Analysis(
    ["src/voicecommander/__main__.py"],
    pathex=["src"],
    # soundcard loads its cffi header definitions from package data at runtime.
    datas=collect_data_files("soundcard"),
    hiddenimports=["keyring.backends.Windows"],
    # numpy stays in: soundcard needs it for the captions loopback capture.
    excludes=[] if nvidia_build else ["librosa", "llvmlite", "numba", "scipy", "sklearn", "torch", "transformers"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VoiceCommander",
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VoiceCommander",
)
