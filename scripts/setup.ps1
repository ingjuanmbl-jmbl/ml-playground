param(
    [string]$EnvironmentPath = ".venv"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ResolvedEnvironmentPath = Join-Path $ProjectRoot $EnvironmentPath

if (-not (Test-Path -LiteralPath $ResolvedEnvironmentPath)) {
    py -3.14 -m venv $ResolvedEnvironmentPath
}

$PythonPath = Join-Path $ResolvedEnvironmentPath "Scripts\python.exe"
$ConfigPath = Join-Path $ResolvedEnvironmentPath "pyvenv.cfg"
if (-not (Test-Path -LiteralPath $ConfigPath) -or -not (Test-Path -LiteralPath $PythonPath)) {
    throw "The environment at '$ResolvedEnvironmentPath' is incomplete. Choose a new path, for example .venv-clean."
}

& $PythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip." }
$InstallTarget = "${ProjectRoot}[dev]"
& $PythonPath -m pip install -e $InstallTarget
if ($LASTEXITCODE -ne 0) { throw "Could not install the project from pyproject.toml." }
& $PythonPath -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency verification failed." }

& $PythonPath -c "import numpy, pandas, plotly, sklearn, streamlit, xgboost; import sklearn.utils; print('Environment imports verified.')"
if ($LASTEXITCODE -ne 0) { throw "Environment import verification failed." }
