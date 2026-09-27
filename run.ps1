Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
    Write-Error "Virtual environment not found. Run setup.ps1 first to create .venv."
    exit 1
}
$HasConfig = $false
foreach ($Item in $args) {
    if (($Item -eq "--config") -or ($Item.StartsWith("--config="))) {
        $HasConfig = $true
    }
}
if ($HasConfig) {
    & ".\.venv\Scripts\python.exe" -m traffic_counter.cli @args
} else {
    Write-Host "Using default config: config.toml"
    & ".\.venv\Scripts\python.exe" -m traffic_counter.cli --config config.toml @args
}
