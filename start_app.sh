#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# One fixed port, so a bookmark (or a Dock web app) always works. LLM_NOTES_SERVICE is set when
# llm_notes_service.sh runs this in the background.
PORT="${LLM_NOTES_PORT:-8510}"
URL="http://localhost:$PORT"
if curl -fsS -m 2 "$URL/_stcore/health" >/dev/null 2>&1; then
    echo "LLM-Notes is already running at $URL"
    [ -z "${LLM_NOTES_SERVICE:-}" ] && open "$URL"
    exit 0
fi

# Large full-corpus ingests open many files (SQLite, LanceDB, PDFs). Raise the limit on macOS.
ulimit -n 65536 2>/dev/null || ulimit -n 10240 2>/dev/null || true

if [ -d "./venv" ]; then
    source ./venv/bin/activate
elif [ -d "./.venv" ]; then
    source ./.venv/bin/activate
else
    echo "Virtual environment not found in $PROJECT_DIR"
    echo "Run ./Installer.sh from the project folder first (it creates venv and installs dependencies)."
    exit 1
fi

for pkg_import in transformers torchvision pydub pyarrow; do
    if ! python3 -c "import ${pkg_import}" 2>/dev/null; then
        echo "Missing dependency: ${pkg_import} (required for indexing/embedding)."
        echo "Run: pip install --requirement requirements.txt --upgrade-strategy eager"
        if [ "${pkg_import}" = "pydub" ]; then
            echo "On Python 3.13+, pydub also needs: pip install audioop-lts"
        fi
        exit 1
    fi
done

echo "Ensuring Ollama is running for Answer with LLM..."
python3 ensure_ollama_ready.py || echo "Warning: Ollama warm-up failed; continuing anyway."

echo "Starting RAG Workspace Manager on $URL..."
# Bind to localhost only — the app has no login and can show full documents. In the background service there's
# no one to show a browser tab to, so it runs headless.
HEADLESS=false
[ -n "${LLM_NOTES_SERVICE:-}" ] && HEADLESS=true
exec streamlit run UI-semantic-search.py --server.address localhost --server.port "$PORT" \
    --server.headless "$HEADLESS"
