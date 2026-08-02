[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ReviewFile,
    [Parameter(Mandatory = $true)][string]$Reviewer,
    [switch]$Push
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw ".venv was not found." }

$Arguments = @("-m", "scripts.dataset_cycle", "approve", $ReviewFile,
    "--reviewer", $Reviewer, "--commit")
if ($Push) { $Arguments += "--push" }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "dataset_cycle approve failed." }
