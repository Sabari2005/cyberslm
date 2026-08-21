"""
Push training inputs to the Modal `cyberslm-data` volume.

    python infra/upload_data.py            # upload anything missing
    python infra/upload_data.py --force    # re-upload everything

Roughly 470 MB total, so this takes a while on a home connection but only has
to happen once. Files already present on the volume with a matching size are
skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VOLUME = "cyberslm-v1-data"

# (local path, name on the volume)
# Ordered smallest-first on purpose. These are multi-hundred-MB transfers over
# a home connection and Modal's client raises ConnectionError on a drop; getting
# the cheap files banked first means a failure on train.bin costs only train.bin.
FILES = [
    (REPO / "tokenizer" / "tokenizer_output" / "tokenizer.model", "tokenizer.model"),
    (REPO / "tokenizer" / "data" / "val.bin", "val.bin"),
    (REPO / "cyberslm_sft" / "data" / "SFT.jsonl", "SFT.jsonl"),
    (REPO / "tokenizer" / "data" / "train.bin", "train.bin"),
]

MAX_ATTEMPTS = 5


def existing() -> dict[str, int]:
    """Map remote filename -> size in bytes, empty if the volume is new."""
    try:
        out = subprocess.run(
            ["modal", "volume", "ls", VOLUME],
            capture_output=True, text=True, timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if out.returncode != 0:
        return {}
    found: dict[str, int] = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 2 and not parts[0].startswith("-"):
            name = parts[0]
            for p in parts[1:]:
                digits = p.replace(",", "").replace("B", "").strip()
                if digits.isdigit():
                    found[name] = int(digits)
                    break
            else:
                found[name] = -1
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-upload even if a same-size file is already there")
    args = ap.parse_args()

    missing_local = [p for p, _ in FILES if not p.exists()]
    if missing_local:
        print("Missing local files:", file=sys.stderr)
        for p in missing_local:
            print(f"  {p}", file=sys.stderr)
        return 1

    remote = {} if args.force else existing()
    total = sum(p.stat().st_size for p, _ in FILES)
    print(f"Volume : {VOLUME}")
    print(f"Payload: {total / 1e6:.0f} MB across {len(FILES)} files\n")

    for local, name in FILES:
        size = local.stat().st_size
        if not args.force and remote.get(name, -1) == size:
            print(f"  skip    {name:<18} ({size / 1e6:.1f} MB, already uploaded)")
            continue
        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"  upload  {name:<18} ({size / 1e6:.1f} MB) "
                  f"attempt {attempt}/{MAX_ATTEMPTS} ...", flush=True)
            r = subprocess.run(
                ["modal", "volume", "put", VOLUME, str(local), name, "--force"],
                timeout=3 * 60 * 60,
            )
            if r.returncode == 0:
                print(f"          {name} OK", flush=True)
                break
            if attempt == MAX_ATTEMPTS:
                print(f"FAILED uploading {name} after {MAX_ATTEMPTS} attempts",
                      file=sys.stderr)
                return r.returncode
            wait = 10 * attempt
            print(f"          dropped; retrying in {wait}s", flush=True)
            time.sleep(wait)

    print("\nDone. Verify with:  modal run infra/modal_app.py::ls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
