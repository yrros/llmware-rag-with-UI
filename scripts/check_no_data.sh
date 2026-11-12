#!/bin/bash
# Stop personal data and secrets reaching GitHub, even if .gitignore is changed or a file is force-added.
#
#   scripts/check_no_data.sh --staged   what's about to be committed (the pre-commit hook runs this)
#   scripts/check_no_data.sh --tracked  every file in the repo (the GitHub check runs this)
#
# It refuses: the app's data (llmware_data/, outbox/, settings, prompts, chats, the actions log), databases and
# vector tables, keys and certificates, and text that looks like an email address or an access key.
# Data belongs in ~/Library/Application Support/LLM-Notes (or $LLM_NOTES_DATA_DIR), never in this folder.
# A false alarm can be committed anyway with: git commit --no-verify
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 2
mode="${1:---staged}"

BLOCKED_PATHS='(^|/)(llmware_data|outbox|chats|backups|lancedb|accounts)/|(^|/)(app_settings|corpus_settings|prompt_library|actions_log|job_status|text_rescan_status|opportunities|tasks)\.json$|\.(db|sqlite3?|lance|p12|pfx|pem|key|mobileconfig|password|ovpn|kdbx)$|(^|/)\.env(\.|$)|(^|/)config\.toml$|(^|/)id_(rsa|ed25519|ecdsa)'
EMAIL='[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,24}'
EMAIL_OK='@(example\.(com|org|net)|users\.noreply\.github\.com|anthropic\.com)|@[0-9]+x\.(png|jpe?g|gif|svg|icns)'
SECRETS='-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35}'

if [ "$mode" = "--staged" ]; then
    files="$(git diff --cached --name-only --diff-filter=ACMR)"
    added="$(git diff --cached -U0 --diff-filter=ACMR --no-color | grep -E '^\+' | grep -vE '^\+\+\+ ')"
elif [ "$mode" = "--tracked" ]; then
    files="$(git ls-files)"
    added="$(git grep -hI -e '' -- . 2>/dev/null)"
else
    echo "Usage: $0 [--staged|--tracked]"; exit 2
fi

problems=0
bad_files="$(printf '%s\n' "$files" | grep -E "$BLOCKED_PATHS")"
if [ -n "$bad_files" ]; then
    echo "✋ These look like your data or keys, which never go to GitHub:"
    printf '   %s\n' $bad_files
    problems=1
fi
emails="$(printf '%s\n' "$added" | grep -oE "$EMAIL" | grep -vE "$EMAIL_OK" | sort -u)"
if [ -n "$emails" ]; then
    echo "✋ Email addresses in the changes (use a placeholder such as name@example.com):"
    printf '   %s\n' $emails
    problems=1
fi
if printf '%s\n' "$added" | grep -qE -- "$SECRETS"; then
    echo "✋ Something that looks like a private key or access token is in the changes."
    problems=1
fi
if [ "$mode" = "--staged" ] && command -v gitleaks >/dev/null 2>&1; then
    gitleaks protect --staged --no-banner --redact >/dev/null 2>&1 || {
        echo "✋ gitleaks found a secret: run 'gitleaks protect --staged -v' to see it."; problems=1; }
fi
if [ "$problems" -ne 0 ]; then
    echo "Nothing was committed. Data belongs in ~/Library/Application Support/LLM-Notes, not in this folder."
    echo "If this is a false alarm: git commit --no-verify"
    exit 1
fi
exit 0
