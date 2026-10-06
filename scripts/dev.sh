#!/usr/bin/env bash
# Lance l'API (rechargement auto) et le serveur Vite de l'UI : http://localhost:5173
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d frontend/node_modules ] || (cd frontend && npm install)
uv run schedule-optimizer serve --reload &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT
cd frontend && npm run dev
