# Submission verification only; experiments are run separately using README.
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot ".venv/Scripts/python.exe"
$projectRuff = Join-Path $PSScriptRoot ".venv/Scripts/ruff.exe"
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw "Missing environment. Provision it with uv sync --frozen --group dev."
}
if (-not (Test-Path -LiteralPath $projectRuff)) {
    throw "Ruff is unavailable in the existing environment."
}
& $projectPython -B -m pytest -p no:cacheprovider tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $projectRuff check --no-cache .
exit $LASTEXITCODE
