"""
Python port of CICADA_FIGURES/run_all_figures.m

Orchestrator: invokes each ported figure script via subprocess so each runs
in its own interpreter (mirrors MATLAB's clearvars-between-scripts isolation).
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "a4_HeatMaps_Combined.py",   # -> Fig_heatmap_figure.pdf
    "a4_HeatMaps_Aggressive_Figs.py",  # -> Fig_heatmaps.pdf
    "a5_OptimizationCurve.py",   # -> Fig_optimization_curves_with_survival.pdf
    "plot_survival_curves_only.py",    # -> Fig_survival_curves_only.pdf
]


def main():
    print("\n================================================")
    print(f"RUNNING {len(SCRIPTS)} FIGURE SCRIPTS")
    print("================================================")

    failures = []
    for i, s in enumerate(SCRIPTS, 1):
        path = os.path.join(HERE, s)
        if not os.path.exists(path):
            print(f"\n[{i}/{len(SCRIPTS)}] SKIP (missing): {s}")
            continue
        print(f"\n[{i}/{len(SCRIPTS)}] {s} ...")
        t0 = time.time()
        try:
            res = subprocess.run(
                [sys.executable, path],
                cwd=HERE,
                check=False,
                capture_output=True,
                text=True,
            )
            dt = time.time() - t0
            if res.returncode == 0:
                print(f"  [OK] {s} ({dt:.1f}s)")
            else:
                print(f"  [FAIL] {s} (rc={res.returncode}, {dt:.1f}s)")
                failures.append(s)
            if res.stdout:
                print(res.stdout.rstrip())
            if res.stderr:
                print(res.stderr.rstrip(), file=sys.stderr)
        except Exception as e:
            print(f"  [FAIL] {s}: {e}")
            failures.append(s)

    print("\n================================================")
    print("DONE" + (f" ({len(failures)} failed: {failures})" if failures else ""))
    print("================================================")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
