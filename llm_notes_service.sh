#!/bin/bash
# LLM-Notes in the background on one fixed port, so one bookmark or a Dock web app always works. It starts only
# when you open it (the Desktop LLM-Notes icon, or `open`), never at login, and stops when you quit the icon
# (right-click it in the Dock > Quit) or run ./stop_app.sh.
#
#   ./llm_notes_service.sh install     build the Desktop LLM-Notes icon (doesn't start anything)
#   ./llm_notes_service.sh open        open it in the browser, starting it first if needed
#   ./llm_notes_service.sh stop        stop it and free its memory (also ./stop_app.sh)
#   ./llm_notes_service.sh restart     restart it, e.g. after updating the app
#   ./llm_notes_service.sh status      is it running and answering?
#   ./llm_notes_service.sh logs        follow its log
#
# The app is started by whatever runs `open`: the LLM-Notes icon (macOS asks once to let it use the Documents
# folder, where the project lives) or Terminal. Set LLM_NOTES_PORT to use a port other than 8510.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LABEL="io.github.yrros.llm-notes"
SUPPORT_DIR="$HOME/Library/Application Support/LLM-Notes"
PID_FILE="$SUPPORT_DIR/llm-notes.pid"
LOG_DIR="$HOME/Library/Logs/LLM-Notes"
LOG="$LOG_DIR/llm-notes.log"
DESKTOP_APP="$HOME/Desktop/LLM-Notes.app"
ICON="$PROJECT_DIR/assets/AppIcon.icns"
PORT="${LLM_NOTES_PORT:-8510}"
URL="http://localhost:$PORT"

is_up() { curl -fsS -m 2 "$URL/_stcore/health" >/dev/null 2>&1; }

# The process `open` started (start_app.sh, which becomes the Streamlit server), if it's still running.
started_pid() {
    local pid
    [[ -f "$PID_FILE" ]] || return 0
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && ps -o command= -p "$pid" 2>/dev/null | grep -qE "start_app.sh|UI-semantic-search.py"; then
        echo "$pid"
    fi
    return 0
}

# The LLM-Notes server on the port, however it was started (this script or ./start_app.sh in Terminal).
app_pids() {
    local pid
    for pid in $(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null); do
        if ps -o command= -p "$pid" 2>/dev/null | grep -q "UI-semantic-search.py"; then
            echo "$pid"
        fi
    done
    return 0
}

wait_up() {
    local seconds="${1:-180}"
    for _ in $(seq 1 "$seconds"); do
        is_up && return 0
        # It gave up already (e.g. no access to the project folder): no point waiting the full time.
        [[ -n "$(started_pid)$(app_pids)" ]] || return 1
        sleep 1
    done
    return 1
}

start_in_background() {
    mkdir -p "$SUPPORT_DIR" "$LOG_DIR"
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') starting on $URL" >> "$LOG"
    cd "$PROJECT_DIR"
    PATH="/opt/homebrew/bin:/usr/local/bin:$PATH" LANG="${LANG:-en_US.UTF-8}" LLM_NOTES_PORT="$PORT" \
        LLM_NOTES_SERVICE=1 nohup /bin/bash "$PROJECT_DIR/start_app.sh" >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
}

open_app() {
    if ! is_up; then
        # A second click while it's still starting just waits for it.
        if [[ -z "$(started_pid)$(app_pids)" ]]; then
            start_in_background
            # Configure > Notifications > Notifications from this app: off sends none, this one included.
            if ! grep -Eq '"notifications_enabled"[[:space:]]*:[[:space:]]*false' \
                    "$PROJECT_DIR/llmware_data/app_settings.json" 2>/dev/null; then
                osascript -e 'display notification "Starting LLM-Notes…" with title "LLM-Notes"' 2>/dev/null || true
            fi
        fi
        if ! wait_up 180; then
            osascript -e "display alert \"LLM-Notes didn't start\" message \"See the log: $LOG

If it says Operation not permitted, allow LLM-Notes in System Settings > Privacy & Security > Files and Folders (Documents folder).\"" \
                2>/dev/null || true
            exit 1
        fi
    fi
    open "$URL"
}

stop_app() {
    local pids
    pids="$( { started_pid; app_pids; } | sort -u | tr '\n' ' ')"
    if [[ -z "${pids// /}" ]]; then
        rm -f "$PID_FILE"
        echo "LLM-Notes wasn't running."
        return 0
    fi
    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null || true
    for _ in $(seq 1 15); do
        [[ -z "$(started_pid)$(app_pids)" ]] && break
        sleep 1
    done
    if [[ -n "$(started_pid)$(app_pids)" ]]; then
        # shellcheck disable=SC2086
        kill -KILL $pids 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    echo "Stopped LLM-Notes."
}

# A stay-open AppleScript app: clicking it opens LLM-Notes (starting it if needed) and it stays in the Dock while
# LLM-Notes runs, so right-click > Quit (or Cmd-Q) stops it. It leaves the Dock by itself if LLM-Notes stops
# some other way. As the app that starts the server, it's what macOS asks about for the Documents folder.
build_desktop_app() {
    local script work
    work="$(mktemp -d)"
    script="$work/LLM-Notes.applescript"
    cat > "$script" <<EOF
property serviceScript : "$PROJECT_DIR/llm_notes_service.sh"
property openedAt : missing value

on run
	my openNotes()
end run

on reopen
	my openNotes()
end reopen

on openNotes()
	-- In the background, so the icon stays responsive while LLM-Notes starts (up to a minute or so).
	do shell script "/bin/bash " & quoted form of serviceScript & " open > /dev/null 2>&1 &"
	set openedAt to current date
end openNotes

on idle
	-- Leave the Dock once LLM-Notes has stopped (stop_app.sh, or it failed to start), after a start-up grace.
	try
		do shell script "/bin/bash " & quoted form of serviceScript & " is-up"
	on error
		if openedAt is not missing value and ((current date) - openedAt) > 240 then quit
	end try
	return 20
end idle

on quit
	try
		do shell script "/bin/bash " & quoted form of serviceScript & " stop > /dev/null 2>&1"
	end try
	continue quit
end quit
EOF
    rm -rf "$DESKTOP_APP"
    osacompile -s -o "$DESKTOP_APP" "$script"
    if [[ -f "$ICON" ]]; then
        # osacompile's asset catalog would override the icon file, so use the LLM-Notes logo alone.
        cp "$ICON" "$DESKTOP_APP/Contents/Resources/applet.icns"
        rm -f "$DESKTOP_APP/Contents/Resources/Assets.car"
        /usr/libexec/PlistBuddy -c "Delete :CFBundleIconName" "$DESKTOP_APP/Contents/Info.plist" 2>/dev/null || true
    fi
    /usr/libexec/PlistBuddy -c "Set :CFBundleName LLM-Notes" "$DESKTOP_APP/Contents/Info.plist" 2>/dev/null || true
    /usr/libexec/PlistBuddy -c "Add :CFBundleIdentifier string $LABEL.launcher" "$DESKTOP_APP/Contents/Info.plist" 2>/dev/null \
        || /usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier $LABEL.launcher" "$DESKTOP_APP/Contents/Info.plist"
    codesign --force --deep --sign - "$DESKTOP_APP" >/dev/null 2>&1 || true  # re-sign after the changes above
    xattr -cr "$DESKTOP_APP" 2>/dev/null || true
    touch "$DESKTOP_APP"
    rm -rf "$work"
    echo "Desktop icon: $DESKTOP_APP (click to open; right-click its Dock icon > Quit to stop)"
}

# Earlier versions ran it as a launchd job, which can't reach a project in the Documents folder.
remove_launchd_job() {
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$SUPPORT_DIR/$LABEL.plist" "$HOME/Library/LaunchAgents/$LABEL.plist"
}

status_app() {
    local started running
    started="$(started_pid)"
    running="$(app_pids)"
    if [[ -n "$running" && -n "$started" ]]; then
        echo "Running:   yes, in the background (pid $started)"
    elif [[ -n "$running" ]]; then
        echo "Running:   yes, from Terminal"
    elif [[ -n "$started" ]]; then
        echo "Running:   starting (pid $started)"
    else
        echo "Running:   no"
    fi
    echo "Answering: $(is_up && echo "yes, $URL" || echo no)"
    echo "Log:       $LOG"
}

case "${1:-}" in
    install) remove_launchd_job; build_desktop_app ;;
    open) open_app ;;
    stop) stop_app ;;
    restart) stop_app >/dev/null; open_app ;;
    status) status_app ;;
    is-up) is_up || [[ -n "$(started_pid)" ]] ;;  # starting counts, so the Dock icon waits for it
    logs) mkdir -p "$LOG_DIR"; touch "$LOG"; tail -n 50 -f "$LOG" ;;
    *) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
