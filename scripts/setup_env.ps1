$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
py -m pip install -r requirements.txt
py -m pytest tests -q
