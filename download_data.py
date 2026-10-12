#!/usr/bin/env python3
"""Download a LibriTTS-R split from ModelScope."""

import argparse

from modelscope.msdatasets import MsDataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a LibriTTS-R split from ModelScope."
    )
    parser.add_argument(
        "--dataset",
        default="modelscope/libritts_r",
        help="ModelScope dataset name (default: modelscope/libritts_r).",
    )
    parser.add_argument(
        "--subset-name",
        default="default",
        help="Dataset subset name (default: default).",
    )
    parser.add_argument(
        "--split",
        default="dev.clean",
        help="Dataset split to download (default: dev.clean).",
    )
    parser.add_argument(
        "--cache-dir",
        help="Optional ModelScope cache directory.",
    )
    args = parser.parse_args()

    load_kwargs = {
        "subset_name": args.subset_name,
        "split": args.split,
    }
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir

    print(f"Loading {args.dataset} ({args.split})")
    dataset = MsDataset.load(args.dataset, **load_kwargs)
    print(f"Download complete: {dataset}")


if __name__ == "__main__":
    main()