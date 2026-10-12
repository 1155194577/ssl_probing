#!/usr/bin/env python3
"""Download the LibriTTS train-clean-100 archive."""

import argparse
from pathlib import Path
from urllib.request import urlopen

URL = "https://openslr.magicdatatech.com/resources/60/train-clean-100.tar.gz"
DEFAULT_OUTPUT = Path("data/train-clean-100.tar.gz")


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
        while chunk := response.read(1024 * 1024):
            output_file.write(chunk)

    print(f"Download complete: {args.output}")


if __name__ == "__main__":
    main()