#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

if [[ "${PRODUCTION:-0}" == "1" ]]; then
  exec gunicorn --bind 0.0.0.0:${PORT:-7860} --workers ${WEB_CONCURRENCY:-1} app:app
else
  exec python app.py
fi
