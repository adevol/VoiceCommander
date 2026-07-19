import os

nvidia_build = os.environ.get("VOICECOMMANDER_BUILD_NVIDIA") == "1"

a = Analysis(
    ["src/voicecommander/__main__.py"],
    pathex=["src"],
    hiddenimports=["keyring.backends.Windows"],
    excludes=[] if nvidia_build else ["librosa", "llvmlite", "numba", "numpy", "scipy", "sklearn", "torch", "transformers"],
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
