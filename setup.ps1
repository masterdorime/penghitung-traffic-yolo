Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $PythonCmd) {
    Write-Error "Python 3.14 is required but the python command was not found on PATH."
    exit 1
}
$VersionText = & python --version 2>&1
if ("$VersionText" -notmatch "Python 3\.14") {
    Write-Error "Python 3.14 is required but found: $VersionText"
    exit 1
}
if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
    & python -m venv .venv
}
$VenvVersion = & ".\.venv\Scripts\python.exe" --version 2>&1
if ("$VenvVersion" -notmatch "Python 3\.14") {
    Write-Error "The existing .venv does not report Python 3.14 (found: $VenvVersion). Delete the .venv directory and rerun setup."
    exit 1
}
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -e ".[dev]"
Write-Output "Setup complete."
Write-Output "Next steps:"
Write-Output "Run calibration still capture: .\\run.ps1 --calibrate"
Write-Output "Run live smoke check: .\\run.ps1 --smoke-seconds 30"
Write-Output "Run normal counter: .\\run.ps1"
