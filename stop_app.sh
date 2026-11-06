#!/bin/bash
# Stop LLM-Notes and free its memory, however it was started (the Desktop icon or ./start_app.sh).
exec /bin/bash "$(cd "$(dirname "$0")" && pwd)/llm_notes_service.sh" stop
