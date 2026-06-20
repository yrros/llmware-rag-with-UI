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
    exit 1
fi

echo "Ensuring Ollama is running for Answer with LLM..."
python3 ensure_ollama_ready.py || echo "Warning: Ollama warm-up failed; continuing anyway."

echo "Starting RAG Workspace Manager..."
exec streamlit run UI-semantic-search.py
