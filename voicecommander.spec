"""PyInstaller build definition."""

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

transcribe_datas, transcribe_binaries, transcribe_hiddenimports = collect_all("transcribe_cpp")
native_datas, native_binaries, native_hiddenimports = collect_all("transcribe_cpp_native")
ort_datas, ort_binaries, ort_hiddenimports = collect_all("onnxruntime")

a = Analysis(
    ["src/voicecommander/__main__.py"],
    pathex=["src"],
    datas=[
        *collect_data_files("soundcard"),
        *transcribe_datas,
        *native_datas,
        *ort_datas,
        *copy_metadata("transcribe-cpp"),
        *copy_metadata("transcribe-cpp-native"),
    ],
    binaries=[*transcribe_binaries, *native_binaries, *ort_binaries],
    hiddenimports=[
        "keyring.backends.Windows",
        *collect_submodules("onnx_asr"),
        *transcribe_hiddenimports,
        *native_hiddenimports,
        *ort_hiddenimports,
    ],
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
