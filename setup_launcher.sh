#!/bin/bash

# Get the absolute path to the directory where this script is located
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCHER_NAME="Launch_Rag_App.command"
LAUNCHER_PATH="$HOME/Desktop/$LAUNCHER_NAME"

echo "Creating launcher at: $LAUNCHER_PATH"

# Write the commands to the .command file
cat <<EOF > "$LAUNCHER_PATH"
#!/bin/bash
cd "$PROJECT_DIR"
# Check if venv exists and activate it
if [ -d "./venv" ]; then
    source ./venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found in $PROJECT_DIR"
    exit 1
fi

echo "Ensuring Ollama is running for Answer with LLM..."
python3 ensure_ollama_ready.py || echo "Warning: Ollama warm-up failed; continuing anyway."

echo "Starting RAG Workspace Manager..."
streamlit run UI-semantic-search.py
EOF

# Make the launcher executable
chmod +x "$LAUNCHER_PATH"

echo "Setup complete! You can now find '$LAUNCHER_NAME' on your Desktop."
