#!/usr/bin/env python3

import argparse
import concurrent.futures
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests


# ============================================================
# CONFIG
# ============================================================

PROFILE_URL = "https://soundgasm.net/u/{username}"

AUDIO_URL_RE = re.compile(
    r'https://soundgasm\.net/u/[^"\'<>\s]+',
    re.IGNORECASE
)

MEDIA_URL_RE = re.compile(
    r'https?://[^"\'<>\s]+?\.(?:m4a|mp3|ogg)(?:\?[^"\'<>\s]*)?',
    re.IGNORECASE
)

VALID_USERNAME_RE = re.compile(
    r'^[A-Za-z0-9_-]+$'
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


# ============================================================
# SESSION
# ============================================================

def make_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


# ============================================================
# UTILITIES
# ============================================================

def clean_url(url):
    """
    Remove HTML-ish trailing characters that sometimes get
    caught by regex.
    """
    return url.rstrip('.,;:)]}\'"')


def safe_filename(name):
    """
    Make a filesystem-safe filename.
    """
    name = unquote(name)

    name = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]',
        "_",
        name
    )

    name = name.strip(" .")

    if not name:
        name = "audio"

    return name


def audio_filename(audio_url):
    """
    Use the final Soundgasm URL component as filename.

    Example:
        /u/foo/mommy-comfort
        -> mommy-comfort.m4a
    """

    parsed = urlparse(audio_url)

    slug = Path(parsed.path).name

    slug = safe_filename(slug)

    return slug


# ============================================================
# PROFILE
# ============================================================

def get_profile(session, username):

    url = PROFILE_URL.format(
        username=username
    )

    print(f"[PROFILE] {username}")

    response = session.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    return response.text


def extract_audio_pages(html, username):

    matches = AUDIO_URL_RE.findall(html)

    unique = set()

    prefix = (
        f"https://soundgasm.net/u/{username}/"
    ).lower()

    for url in matches:

        url = clean_url(url)

        # We only want this user's audio pages.
        if not url.lower().startswith(prefix):
            continue

        # Avoid accidentally treating the profile itself
        # as an audio page.
        remainder = url[len(prefix):]

        if not remainder or "/" in remainder:
            continue

        unique.add(url)

    return sorted(unique)


# ============================================================
# AUDIO PAGE
# ============================================================

def get_media_url(session, audio_page_url):

    response = session.get(
        audio_page_url,
        timeout=30
    )

    response.raise_for_status()

    html = response.text

    matches = MEDIA_URL_RE.findall(html)

    if not matches:
        return None

    return clean_url(matches[0])


# ============================================================
# DOWNLOAD
# ============================================================

def download_audio(
    session,
    username,
    audio_page_url,
    output_dir
):

    try:

        filename = audio_filename(
            audio_page_url
        )

        # Find the actual media URL first.
        media_url = get_media_url(
            session,
            audio_page_url
        )

        if not media_url:
            print(
                f"  [NO MEDIA] {audio_page_url}"
            )
            return False

        # Determine actual extension.
        parsed = urlparse(media_url)

        path = parsed.path.lower()

        if path.endswith(".mp3"):
            extension = ".mp3"
        elif path.endswith(".ogg"):
            extension = ".ogg"
        else:
            extension = ".m4a"

        filename += extension

        user_dir = (
            Path(output_dir) /
            safe_filename(username)
        )

        user_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = user_dir / filename

        # Skip existing files.
        if output_file.exists():
            print(
                f"  [SKIP] {output_file}"
            )
            return True

        print(
            f"  [DOWNLOAD] {audio_page_url}"
        )

        print(
            f"             -> {output_file}"
        )

        with session.get(
            media_url,
            stream=True,
            timeout=60
        ) as response:

            response.raise_for_status()

            with open(
                output_file,
                "wb"
            ) as f:

                for chunk in response.iter_content(
                    chunk_size=1024 * 128
                ):

                    if chunk:
                        f.write(chunk)

        print(
            f"  [DONE] {output_file}"
        )

        return True

    except Exception as e:

        print(
            f"  [ERROR] {audio_page_url}: {e}"
        )

        return False


# ============================================================
# USER
# ============================================================

def process_user(
    username,
    output_dir,
    workers
):

    session = make_session()

    try:

        html = get_profile(
            session,
            username
        )

        audio_pages = extract_audio_pages(
            html,
            username
        )

        print(
            f"  Found {len(audio_pages)} "
            f"audio pages"
        )

        if not audio_pages:
            return 0, 0

        successful = 0

        # ----------------------------------------------------
        # Sequential
        # ----------------------------------------------------

        if workers <= 1:

            for audio_page in audio_pages:

                if download_audio(
                    session,
                    username,
                    audio_page,
                    output_dir
                ):
                    successful += 1

        # ----------------------------------------------------
        # Concurrent
        # ----------------------------------------------------

        else:

            # Each worker gets its own Session.
            def worker(audio_page):
                worker_session = make_session()

                return download_audio(
                    worker_session,
                    username,
                    audio_page,
                    output_dir
                )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers
            ) as executor:

                futures = [
                    executor.submit(
                        worker,
                        audio_page
                    )
                    for audio_page in audio_pages
                ]

                for future in concurrent.futures.as_completed(
                    futures
                ):

                    try:
                        if future.result():
                            successful += 1
                    except Exception as e:
                        print(
                            f"  [WORKER ERROR] {e}"
                        )

        return len(audio_pages), successful

    except Exception as e:

        print(
            f"[USER ERROR] {username}: {e}"
        )

        return 0, 0

    finally:
        session.close()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Download public Soundgasm audio for usernames."
    )

    parser.add_argument(
        "usernames",
        nargs="?",
        default="usernames.txt",
        help="File containing one username per line"
    )

    parser.add_argument(
        "--output",
        default="audios",
        help="Output directory (default: audios)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent audio downloads per user (default: 1)"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Read usernames
    # --------------------------------------------------------

    username_file = Path(
        args.usernames
    )

    if not username_file.exists():

        print(
            f"ERROR: {username_file} does not exist"
        )

        sys.exit(1)

    usernames = []

    with username_file.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            username = line.strip()

            if not username:
                continue

            # Allow comments.
            if username.startswith("#"):
                continue

            if not VALID_USERNAME_RE.fullmatch(
                username
            ):
                print(
                    f"[SKIP INVALID USERNAME] "
                    f"{username}"
                )
                continue

            usernames.append(username)

    # Deduplicate usernames.
    usernames = list(
        dict.fromkeys(usernames)
    )

    if not usernames:

        print("No usernames found.")
        sys.exit(0)

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    print("=" * 70)
    print("SOUNDGASM DOWNLOADER")
    print("=" * 70)

    print(
        f"Users   : {len(usernames)}"
    )

    print(
        f"Workers : {args.workers}"
    )

    print(
        f"Output  : {args.output}"
    )

    print("=" * 70)

    total_pages = 0
    total_success = 0

    for i, username in enumerate(
        usernames,
        start=1
    ):

        print()
        print(
            f"[{i}/{len(usernames)}] "
            f"{username}"
        )

        pages, successful = process_user(
            username,
            args.output,
            args.workers
        )

        total_pages += pages
        total_success += successful

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(
        f"Users processed : {len(usernames)}"
    )

    print(
        f"Audio pages     : {total_pages}"
    )

    print(
        f"Downloaded/skip : {total_success}"
    )

    print(
        f"Output          : {args.output}/"
    )


if __name__ == "__main__":
    main()