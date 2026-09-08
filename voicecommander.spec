"""PyInstaller build definition."""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

a = Analysis(
    ["src/voicecommander/__main__.py"],
    pathex=["src"],
    datas=collect_data_files("soundcard")
    + [("src/voicecommander/prompts/*.yaml", "voicecommander/prompts")],
    # OpenRouter loads its API resources and response models lazily.
    hiddenimports=["keyring.backends.Windows"] + collect_submodules("openrouter"),
    excludes=["librosa", "llvmlite", "numba", "scipy", "sklearn", "torch", "transformers"],
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
