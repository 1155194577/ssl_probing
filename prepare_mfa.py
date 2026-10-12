#!/usr/bin/env python3
"""Prepare LibriTTS audio and transcripts for Montreal Forced Aligner."""

import argparse
import tarfile
from pathlib import Path


def extract_archive(archive_path: Path, corpus_dir: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        destination = corpus_dir.resolve()
        for member in archive.getmembers():
            member_path = (corpus_dir / member.name).resolve()
            if destination not in member_path.parents:
                raise ValueError(f"Unsafe archive path: {member.name}")
        archive.extractall(corpus_dir)


def create_lab_files(corpus_dir: Path) -> int:
    transcript_files = sorted(corpus_dir.rglob("*.trans.tsv"))
    if not transcript_files:
        raise FileNotFoundError(f"No LibriTTS *.trans.tsv files found in {corpus_dir}")

    created = 0
    for transcript_path in transcript_files:
        with transcript_path.open(encoding="utf-8") as transcript_file:
            for line_number, line in enumerate(transcript_file, start=1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 3:
                    raise ValueError(
                        f"Expected 3 tab-separated fields in "
                        f"{transcript_path}:{line_number}"
                    )

                utterance_id, _, normalized_text = fields
                wav_path = transcript_path.parent / f"{utterance_id}.wav"
                if not wav_path.exists():
                    raise FileNotFoundError(f"Missing audio file: {wav_path}")

                lab_path = wav_path.with_suffix(".lab")
                lab_path.write_text(normalized_text.strip() + "\n", encoding="utf-8")
                created += 1

    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare LibriTTS for Montreal Forced Aligner."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="LibriTTS .tar.gz archive to extract.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/libritts_mfa"),
        help="Corpus directory (default: data/libritts_mfa).",
    )
    args = parser.parse_args()

    if args.archive:
        extract_archive(args.archive, args.corpus_dir)

    created = create_lab_files(args.corpus_dir)
    print(f"Prepared {created} WAV/LAB pairs in {args.corpus_dir}")


if __name__ == "__main__":
    main()