"""Run the whole analysis in order: data summaries, verification, controls, local model, checks, figures.

Usage
-----
python src/run_all.py --project <project root> [--rice-daily <file>] [--out <dir>]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
NEEDS_INPUT = ("run_data_summaries.py", "run_reanalysis.py", "run_controls.py")
STEPS = NEEDS_INPUT + ("run_local_model.py", "run_local_checks.py", "render_figures.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--rice-daily", type=Path)
    ap.add_argument("--out", type=Path, default=SRC.parent)
    args = ap.parse_args()
    (args.out / "tables").mkdir(parents=True, exist_ok=True)
    for step in STEPS:
        cmd = [sys.executable, str(SRC / step), "--out", str(args.out)]
        if step in NEEDS_INPUT:
            cmd += ["--project", str(args.project)]
            if args.rice_daily:
                cmd += ["--rice-daily", str(args.rice_daily)]
        start = time.time()
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        print(f"{step}: {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
