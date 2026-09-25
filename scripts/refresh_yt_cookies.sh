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
DROPLET="${T2A_DROPLET:-root@139.59.33.163}"
TMP="$(mktemp -t ytcookies).txt"
trap 'rm -f "$TMP"' EXIT

echo "exporting cookies from $BROWSER (Keychain may prompt) ..."
yt-dlp --cookies-from-browser "$BROWSER" --cookies "$TMP" --simulate --quiet \
       "https://www.youtube.com/watch?v=aqz-KE-bpKQ" >/dev/null

lines=$(grep -c . "$TMP" || true)
yt=$(grep -c "youtube.com" "$TMP" || true)
if [ "${yt:-0}" -lt 5 ]; then
  echo "refused: jar has only $yt youtube.com entries — the export did not capture a session" >&2
  exit 1
fi
echo "jar looks valid: $lines lines, $yt youtube.com entries"

scp -q "$TMP" "$DROPLET:/root/t2a/cookies.txt"
ssh "$DROPLET" 'chmod 600 /root/t2a/cookies.txt'

# prove it authenticates before restarting anything, so a dead jar is never installed silently
if ssh "$DROPLET" 'cd /root/t2a && ./venv/bin/yt-dlp --cookies cookies.txt --simulate --print "%(title).30s" \
      "https://www.youtube.com/watch?v=aqz-KE-bpKQ" 2>/dev/null | grep -q .'; then
  echo "verified: the droplet can authenticate with the new jar"
  ssh "$DROPLET" 'systemctl restart t2a-yt'
  echo "t2a-yt restarted; backoff penalty cleared"
else
  echo "the new jar did not authenticate on the droplet — leaving t2a-yt alone" >&2
  exit 1
fi
