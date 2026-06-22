#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

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

echo "Starting RAG Workspace Manager..."
exec streamlit run UI-semantic-search.py
