# Create .venv with Python 3.11 and install the locked dependencies (CPU PyTorch wheel index).
# Usage: powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1 [-Python <path to python 3.11>]
param([string]$Python = "py")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$pyArgs = @()
if ($Python -eq "py") { $pyArgs = @("-3.11") }
& $Python @pyArgs -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"
if ($LASTEXITCODE -ne 0) { throw "Python 3.11 required" }
if (Test-Path .venv) { throw ".venv exists; remove it or activate it" }
& $Python @pyArgs -m venv .venv
& .venv\Scripts\python.exe -m pip install --upgrade "pip<26"
& .venv\Scripts\python.exe -m pip install --require-virtualenv -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
Write-Output "done: .venv\Scripts\Activate.ps1; python run.py check --code-only"
