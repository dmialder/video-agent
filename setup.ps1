param([string]$Python='python')
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
$env:UV_CACHE_DIR=Join-Path $PSScriptRoot '.cache/uv'
uv venv --python $Python .venv
if ($LASTEXITCODE -ne 0) { throw 'Use Python 3.12, e.g. -Python C:/Python312/python.exe' }
uv pip install --python .venv/Scripts/python.exe torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
$dependencyFile = if (Test-Path -LiteralPath 'requirements.lock.txt') { 'requirements.lock.txt' } else { 'requirements.txt' }
uv pip install --python .venv/Scripts/python.exe -r $dependencyFile
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& .venv/Scripts/python.exe scripts/prepare_assets.py
if ($LASTEXITCODE -ne 0) { throw 'Asset preparation failed' }
& .venv/Scripts/python.exe scripts/download_model.py
if ($LASTEXITCODE -ne 0) { throw 'Model download failed' }
& .venv/Scripts/python.exe run.py index
