param(
    [switch]$Development
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepositoryRoot

$Targets = @(
    ".",
    ".\tmc-matching",
    ".\cbi",
    ".\nvta-taplite-workflow",
    ".\corridor-performance-dashboard"
)

foreach ($Target in $Targets) {
    if ($Development) {
        python -m pip install -e "$Target[dev]"
    }
    else {
        python -m pip install -e $Target
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Editable install failed: $Target"
    }
}

python .\nvta-taplite-workflow\setup\install_pypi_prerelease.py
if ($LASTEXITCODE -ne 0) {
    throw "Pinned TAPlite installation failed."
}

python .\nvta-taplite-workflow\setup\verify_taplite_contract.py
if ($LASTEXITCODE -ne 0) {
    throw "TAPlite native contract verification failed."
}

Write-Host "CBI-TAPlite environment is ready."
