#!/bin/bash
set -euo pipefail

# Reset the RAG environment: delete indexes, vector tables, histories and temp files so the
# app starts clean (e.g. after an interrupted rebuild leaves the database unusable).
# Downloaded models (llmware_data/model_repo, ~/.cache/huggingface) and the venv are kept.
#
# Usage: ./reset_rag.sh [--yes] [--settings] [--dry-run]
#   --yes       don't ask for confirmation
#   --settings  also delete app_settings.json, corpus_settings.json and prompt_library.json
#               (corpus root, corpus prompts, prompt library, per-corpus models)
#   --dry-run   only list what would be deleted

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
DATA_DIR="$PROJECT_DIR/llmware_data"

ASSUME_YES=0
RESET_SETTINGS=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y) ASSUME_YES=1 ;;
        --settings) RESET_SETTINGS=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '4,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $arg (see --help)"; exit 1 ;;
    esac
done

if [ ! -d "$DATA_DIR" ]; then
    echo "Nothing to reset: $DATA_DIR does not exist."
    exit 0
fi

TARGETS=(
    "accounts"          # SQLite text index (sqlite_llmware.db), library folders, Chroma files
    "lancedb"           # vector tables
    "tmp"               # parser scratch files
    "parser_history"
    "query_history"
    "prompt_history"
    "job_status.json"   # last scan results shown in the sidebar
)
if [ "$RESET_SETTINGS" -eq 1 ]; then
    TARGETS+=("app_settings.json" "corpus_settings.json" "prompt_library.json")
fi

EXISTING=()
for target in "${TARGETS[@]}"; do
    if [ -e "$DATA_DIR/$target" ]; then
        EXISTING+=("$target")
    fi
done

if [ "${#EXISTING[@]}" -eq 0 ]; then
    echo "Nothing to reset: no indexes or history found in $DATA_DIR."
    exit 0
fi

echo "This will delete from $DATA_DIR:"
for target in "${EXISTING[@]}"; do
    size="$(du -sh "$DATA_DIR/$target" 2>/dev/null | awk '{print $1}')"
    echo "  - $target (${size:-?})"
done
echo "Kept: model_repo/, ~/.cache/huggingface, venv/"
if [ "$RESET_SETTINGS" -eq 0 ]; then
    echo "Kept: app_settings.json, corpus_settings.json, prompt_library.json (use --settings to delete them too)"
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run — nothing deleted."
    exit 0
fi

# Deleting the database under a running app would leave it holding stale handles.
if pgrep -f "streamlit run UI-semantic-search.py" >/dev/null 2>&1; then
    echo "The app is still running. Stop it first (Ctrl+C in its terminal, or quit LLM-Notes), then re-run."
    exit 1
fi

if [ "$ASSUME_YES" -eq 0 ]; then
    read -r -p "Continue? [y/N] " reply
    case "$reply" in
        [yY]|[yY][eE][sS]) ;;
        *) echo "Cancelled."; exit 0 ;;
    esac
fi

for target in "${EXISTING[@]}"; do
    rm -rf "${DATA_DIR:?}/$target"
done

# Empty placeholders stop the app re-copying old data from ~/llmware_data on next start.
mkdir -p "$DATA_DIR/accounts" "$DATA_DIR/lancedb"

echo "Reset complete. Start the app with ./start_app.sh — the Getting started guide will walk you through re-indexing."
