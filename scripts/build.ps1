$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

uv sync --group build
uv run python -m unittest discover -s tests
uv run python -m PyInstaller --clean --noconfirm voicecommander.spec

$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (-not (Test-Path -LiteralPath $iscc)) {
    throw "Inno Setup 6 was not found at $iscc"
}
$source = (Resolve-Path "dist\VoiceCommander").Path
& $iscc "/DSourceDir=$source" "installer\VoiceCommander.iss"
