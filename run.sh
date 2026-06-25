#!/usr/bin/env bash
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn server:app --port 7860 --reload
