"""
Pull trained checkpoints down from the Modal `cyberslm-runs` volume.

    python infra/download_model.py                 # base best.pt -> runs/base/
    python infra/download_model.py --run sft       # SFT output   -> runs/sft/
    python infra/download_model.py --all           # everything on the volume
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VOLUME = "cyberslm-v1-runs"


def verify(local: Path) -> bool:
    """
    Confirm a downloaded checkpoint actually deserialises.

    A partially written .pt is indistinguishable from a complete one by size
    alone -- the file can already show its final byte count while the writer
    still holds it open. Loading it is the only honest check, and the failure
    it prevents is a corrupt-archive traceback in the middle of evaluation:

        PytorchStreamReader failed reading file data/277:
        invalid header or archive is corrupted
    """
    try:
        import torch
    except ImportError:
        print("  (torch unavailable; skipping integrity check)")
        return True
    try:
        payload = torch.load(local, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  CORRUPT: {local.name} -> {type(exc).__name__}: {str(exc)[:120]}",
              file=sys.stderr)
        return False
    if isinstance(payload, dict):
        step = payload.get("step")
        vl = payload.get("val_loss")
        n = len(payload.get("model_state", {}))
        print(f"  verified: {local.name}  step={step}  val_loss={vl}  tensors={n}")
    else:
        print(f"  verified: {local.name} (bare state dict)")
    return True


def pull(remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {remote}  ->  {local.relative_to(REPO)}", flush=True)
    r = subprocess.run(
        ["modal", "volume", "get", VOLUME, remote, str(local), "--force"],
        timeout=3 * 60 * 60,
    )
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="base", help="run directory on the volume")
    ap.add_argument("--all", action="store_true", help="pull the whole run directory")
    args = ap.parse_args()

    out = REPO / "runs" / args.run
    print(f"Volume : {VOLUME}")
    print(f"Target : {out}\n")

    if args.all:
        ok = pull(args.run, out)
    else:
        ok = pull(f"{args.run}/best.pt", out / "best.pt")
        if ok and (out / "best.pt").exists():
            ok = verify(out / "best.pt")
        # model.pt is what the SFT stage writes; ignore failure when absent.
        if pull(f"{args.run}/best/model.pt", out / "best" / "model.pt"):
            if (out / "best" / "model.pt").exists():
                verify(out / "best" / "model.pt")

    if not ok:
        print("\nDownload failed. Check what exists with:", file=sys.stderr)
        print("  modal run infra/modal_app.py::ls", file=sys.stderr)
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
