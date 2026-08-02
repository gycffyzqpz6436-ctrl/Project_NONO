[CmdletBinding()]
param([switch]$OpenInNotepad)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw ".venv was not found. Create it and install the required dependencies."
}

& $Python -c "import yaml, rapidfuzz"
if ($LASTEXITCODE -ne 0) {
    throw "Dependencies are missing. Run .venv\Scripts\python -m pip install -r requirements-generation.txt"
}

$Output = & $Python -m scripts.dataset_cycle prepare --count 50
if ($LASTEXITCODE -ne 0) { throw "dataset_cycle prepare failed." }
$Output | Write-Host

$ReviewLine = $Output | Where-Object { $_ -match "Review file:" } | Select-Object -Last 1
$InstructionLine = $Output | Where-Object { $_ -match "Instructions:" } | Select-Object -Last 1
Write-Host $ReviewLine
Write-Host $InstructionLine
if ($OpenInNotepad -and $ReviewLine) {
    $ReviewFile = ($ReviewLine -replace "^.*Review file:\s*", "").Trim()
    Start-Process notepad.exe -ArgumentList $ReviewFile
}
