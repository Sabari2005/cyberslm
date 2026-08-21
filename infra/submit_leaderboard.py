"""
Open a pull request on the Open SLM Leaderboard Space with our results.

    python infra/submit_leaderboard.py --results runs/reports/leaderboard.json --dry-run
    python infra/submit_leaderboard.py --results runs/reports/leaderboard.json

The Space is a static index.html holding a `MODELS` array and an `ORGS` map.
This inserts one entry per model at the bottom of MODELS (the file says
"ADD NEW MODELS TO BOTTOM OF LIST") plus an ORGS entry, then opens a PR via
the Hub API.

--dry-run prints the exact diff and touches nothing remote. Always run it first:
a leaderboard PR is public and the maintainers re-verify every number, so a
malformed entry or a wrong score costs their time as well as yours.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SPACE = "AxiomicLabs/Open_SLM_Leaderboard"

ORG_KEY = "sabari2005"
ORG_ENTRY = (
    "    sabari2005:  { name: 'Sabari',       "
    "chartColor: 'rgba(34, 197, 94, 0.70)',   chartBorder: '#22c55e',  "
    "url: 'https://huggingface.co/sabari2005' },"
)


def model_line(m: dict, name: str | None = None) -> str:
    """Render one MODELS entry in the file's existing style."""
    parts = [
        f"name: {(name or m['name'])!r}",
        f"org: {ORG_KEY!r}",
        f"params: {m['params']}",
        f"paramsDisplay: {m['paramsDisplay']!r}",
        f"arc: {m['arc']:.2f}",
        f"hellaswag: {m['hellaswag']:.2f}",
        f"piqa: {m['piqa']:.2f}",
        f"arcChall: {m['arcChall']:.2f}",
    ]
    if m.get("arithmark3") is not None:
        parts.append(f"arithmark3: {m['arithmark3']:.2f}")
    if m.get("arithmark2") is not None:
        parts.append(f"arithmark2: {m['arithmark2']:.2f}")
    parts.append("links: { card: " + repr(m["card"]) + " }")
    return "    { " + ", ".join(parts) + " },"


def patch(html: str, models: list[dict]) -> str:
    """
    Replace the existing CyberSLM rows in place.

    The project already has CyberSLM-33M-Base/Instruct on the board, pointing at
    the pre-bug-fix July checkpoints. Appending new rows would leave four
    near-identical CyberSLM entries for one 33.5M project, so each row is
    rewritten to point at the retrained repo and carry the new scores.
    """
    # The sabari2005 ORGS entry already exists (the project has rows on the
    # board already), so only the MODELS rows need rewriting.
    if f"{ORG_KEY}:" not in html:
        raise RuntimeError(f"ORGS has no {ORG_KEY} key; add it before submitting")

    for m in models:
        target = m["replaces"]
        pattern = re.compile(
            r"^\s*\{ name: '" + re.escape(target) + r"',.*$", re.M)
        found = pattern.findall(html)
        if len(found) != 1:
            raise RuntimeError(
                f"expected exactly one row named {target!r}, found {len(found)}")
        html = pattern.sub(model_line(m, name=target).rstrip(), html, count=1)
    return html


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="JSON produced by collect_results.py")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--space", default=SPACE)
    args = ap.parse_args()

    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    models = payload["models"]

    from huggingface_hub import HfApi
    api = HfApi()
    path = api.hf_hub_download(args.space, "index.html", repo_type="space")
    html = Path(path).read_text(encoding="utf-8")
    new = patch(html, models)

    import difflib
    diff = list(difflib.unified_diff(
        html.split(chr(10)), new.split(chr(10)),
        fromfile="index.html (current)", tofile="index.html (proposed)",
        lineterm="", n=0))
    print("=" * 74)
    print("  Proposed change")
    print("=" * 74)
    for line in diff:
        print(line[:200])
    print("=" * 74)
    print(f"  size {len(html):,} -> {len(new):,} bytes")
    changed = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    print(f"  {removed} line(s) replaced by {changed}")
    if changed != len(models) or removed != len(models):
        print(f"  WARNING: expected exactly {len(models)} replaced lines", file=sys.stderr)

    if args.dry_run:
        print("\n  --dry-run: nothing was uploaded.")
        return 0

    from huggingface_hub import CommitOperationAdd
    names = ", ".join(m["name"] for m in models)
    res = api.create_commit(
        repo_id=args.space,
        repo_type="space",
        operations=[CommitOperationAdd("index.html", new.encode("utf-8"))],
        commit_message=f"Add {names}",
        commit_description=payload.get("pr_body", ""),
        create_pr=True,
    )
    print(f"\n  PR opened: {res.pr_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
