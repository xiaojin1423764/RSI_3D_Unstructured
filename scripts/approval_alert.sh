#!/usr/bin/env bash
set -euo pipefail

# Notify locally before an operation needs user approval.  Desktop audio is
# optional; remote terminals receive two separated BELs and an OSC 9 notice.
if command -v canberra-gtk-play >/dev/null 2>&1; then
    canberra-gtk-play --id=message-new-instant >/dev/null 2>&1 &
    sleep 0.25
    canberra-gtk-play --id=message-new-instant >/dev/null 2>&1 &
else
    printf '\033]9;Codex approval required\a\a' >&2
    sleep 0.25
    printf '\a' >&2
    printf 'Approval required\n' >&2
fi
