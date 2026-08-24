"""Download/verify the VGGT checkpoint in a CPU/free environment."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path


URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--minimum-bytes", type=int, default=4_000_000_000)
    args = parser.parse_args()
    output = args.output.resolve()

    if output.is_file() and output.stat().st_size >= args.minimum_bytes:
        digest = sha256(output)
        if args.expected_sha256 and digest.lower() != args.expected_sha256.lower():
            raise SystemExit(f"Existing checkpoint hash mismatch: {digest}")
        print(f"Reusing checkpoint: {output} bytes={output.stat().st_size} sha256={digest}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    print(f"Downloading {URL} to {partial}")
    urllib.request.urlretrieve(URL, partial)
    if partial.stat().st_size < args.minimum_bytes:
        raise SystemExit(f"Downloaded checkpoint is unexpectedly small: {partial.stat().st_size}")
    digest = sha256(partial)
    if args.expected_sha256 and digest.lower() != args.expected_sha256.lower():
        raise SystemExit(f"Downloaded checkpoint hash mismatch: {digest}")
    os.replace(partial, output)
    print(f"Cached checkpoint: {output} bytes={output.stat().st_size} sha256={digest}")


if __name__ == "__main__":
    main()
