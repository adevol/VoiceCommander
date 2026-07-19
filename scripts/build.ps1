param([switch]$Nvidia)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$edition = if ($Nvidia) { "nvidia" } else { "standard" }
if ($Nvidia) {
    uv sync --extra nvidia --group build
    $env:VOICECOMMANDER_BUILD_NVIDIA = "1"
} else {
    uv sync --group build
    $env:VOICECOMMANDER_BUILD_NVIDIA = "0"
}
uv run python -m unittest discover -s tests
uv build
uv run python -m PyInstaller --clean --noconfirm --distpath "dist\$edition" --workpath "build\$edition" voicecommander.spec

$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (-not (Test-Path -LiteralPath $iscc)) {
    throw "Inno Setup 6 was not found at $iscc"
}
$source = (Resolve-Path "dist\$edition\VoiceCommander").Path
& $iscc "/DSourceDir=$source" "/DEdition=$edition" "installer\VoiceCommander.iss"
