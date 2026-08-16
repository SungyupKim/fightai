#!/bin/bash
# Launches the 3D fighter viewer. Run directly in your own terminal:
#   ./scripts/run_view3d.sh
cd "$(dirname "$0")"
DISPLAY="${DISPLAY:-:0}" ../.venv/bin/python view3d.py
