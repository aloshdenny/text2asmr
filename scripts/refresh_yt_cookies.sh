#!/bin/bash
# Refresh the YouTube session the droplet's downloader uses.
#
# Must run on the Mac: the cookie jar comes from the user's own browser, and it goes only to the user's own
# droplet. The jar is never printed, never committed, and the local copy is deleted immediately after the
# copy succeeds. YouTube rotates these as the browser is used, so the session lasts ~5 h and this has to be
# re-run; a private window that is closed afterwards lasts longest.
#
#   scripts/refresh_yt_cookies.sh [browser]     # browser defaults to chrome
set -euo pipefail

BROWSER="${1:-chrome}"
# PATH order differs under launchd: ~/.local/bin holds a yt-dlp bound to python3.9, which yt-dlp refuses to
# run on. Pin the working binary rather than trusting whatever the environment resolves first.
YTDLP="${T2A_YTDLP:-/opt/homebrew/bin/yt-dlp}"
[ -x "$YTDLP" ] || YTDLP="$(command -v yt-dlp)"
DROPLET="${T2A_DROPLET:-root@139.59.33.163}"
TMP="$(mktemp -t ytcookies).txt"
trap 'rm -f "$TMP"' EXIT

echo "exporting cookies from $BROWSER (Keychain may prompt) ..."
"$YTDLP" --cookies-from-browser "$BROWSER" --cookies "$TMP" --simulate --quiet \
       "https://www.youtube.com/watch?v=aqz-KE-bpKQ" >/dev/null

lines=$(grep -c . "$TMP" || true)
yt=$(grep -c "youtube.com" "$TMP" || true)
if [ "${yt:-0}" -lt 5 ]; then
  echo "refused: jar has only $yt youtube.com entries — the export did not capture a session" >&2
  exit 1
fi
echo "jar looks valid: $lines lines, $yt youtube.com entries"

# Stage first, verify there, and only then move into place. Installing before verifying means one dead
# export replaces a jar that still worked -- which is exactly what happened on the 22:29 unattended run.
scp -q "$TMP" "$DROPLET:/root/t2a/cookies.staged"
ssh "$DROPLET" 'chmod 600 /root/t2a/cookies.staged'

if ssh "$DROPLET" 'cd /root/t2a && ./venv/bin/yt-dlp --cookies cookies.staged --simulate --print "%(title).30s" \
      "https://www.youtube.com/watch?v=aqz-KE-bpKQ" 2>/dev/null | grep -q .'; then
  ssh "$DROPLET" 'mv /root/t2a/cookies.staged /root/t2a/cookies.txt && systemctl restart t2a-yt'
  echo "verified and installed; t2a-yt restarted, backoff penalty cleared"
else
  ssh "$DROPLET" 'rm -f /root/t2a/cookies.staged'
  echo "the exported jar does not authenticate — the existing cookies.txt was left untouched" >&2
  echo "(log in to YouTube in a private window, then re-run this script)" >&2
  exit 1
fi
