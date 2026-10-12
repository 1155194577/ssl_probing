#!/usr/bin/env python3
"""Download the LibriTTS train-clean-100 archive."""

import argparse
import sys
import time
from pathlib import Path
from urllib.request import urlopen

URL = "https://openslr.magicdatatech.com/resources/60/train-clean-100.tar.gz"
DEFAULT_OUTPUT = Path("data/train-clean-100.tar.gz")


def format_size(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024


def show_progress(downloaded: int, total: int | None, started_at: float) -> None:
    elapsed = max(time.monotonic() - started_at, 0.001)
    speed = downloaded / elapsed
    if total:
        percent = downloaded / total * 100
        status = f"{percent:6.2f}% {format_size(downloaded)} / {format_size(total)}"
    else:
        status = f"{format_size(downloaded)} downloaded"
    sys.stdout.write(
        f"\r{status} at {format_size(int(speed))}/s"
    )
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the LibriTTS train-clean-100 archive."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output archive path (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Output file already exists: {args.output}")

    print(f"Downloading {URL}")
    print(f"Saving to {args.output}")
    with urlopen(URL) as response, args.output.open("wb") as output_file:
        content_length = response.headers.get("Content-Length")
        total = int(content_length) if content_length else None
        downloaded = 0
        started_at = time.monotonic()
        while chunk := response.read(1024 * 1024):
            output_file.write(chunk)
            downloaded += len(chunk)
            show_progress(downloaded, total, started_at)

    print()
    print(f"Download complete: {args.output}")


if __name__ == "__main__":
    main()