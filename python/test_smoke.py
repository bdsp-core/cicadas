"""Smoke tests for the Python port — quick sanity checks that catch syntax
errors, import-graph problems, and obviously-wrong outputs without running
the full (~hours-long) pipeline.

Run:
    python test_smoke.py

Exits with 0 on success, nonzero on failure.
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODULES_TO_IMPORT = [
    "fcnGeneratePatientParameters",
    "fcn_generateStochasticTrajectories",
    "fcn_generateTrajectory",
    "fcnSimulate_N_Patients",
    "fcnSimulate_DoseChanging",
    "fcnSimulate_StructuredDoses",
    "fcnBiasedAssignmentProb",
    "fcnDiseaseModelDiagnostic",
    "fcnEstimateDeathParms",
    "fcnEstimateKe_Standalone",
    "fcn_EstimateKe_WithBiasCorrection",
    "fcnEstimatePKPD_FixedKe_Optimized",
    "fcnEstimatePKPD_StateSpaceMixedEffects_v2",
    "fcnEstimateParmsL",
    "fcnEstimateParmsPKPD",
    "fcnGetPKPD_parms_est",
    "fcnPlotKM",
    "fcn_bootstrapBySID",
    "fcn_estimate_parmsL",
    "fcn_kaplanMeier",
    "fcnSingleSwimmerPlot_v4",
]


def test_imports() -> int:
    failures = 0
    for mod in MODULES_TO_IMPORT:
        try:
            importlib.import_module(mod)
            print(f"  ok  {mod}")
        except Exception as e:
            failures += 1
            print(f"  FAIL {mod}: {e}")
            traceback.print_exc()
    return failures


def test_tiny_simulation() -> int:
    """Run a tiny N=5 simulation to verify the simulators produce sane output."""
    import numpy as np

    np.random.seed(0)

    from fcnGeneratePatientParameters import fcnGeneratePatientParameters
    from fcn_generateStochasticTrajectories import fcnGenerateStochasticTrajectories
    from fcnSimulate_N_Patients import fcnSimulate_N_Patients
    from fcnSimulate_DoseChanging import fcnSimulate_DoseChanging

    failures = 0
    try:
        N = 5
        parmsL = np.array([0.25, 1, 0.15, 0.05, 0.15, 0.03, 40])
        dt = 2
        t = np.arange(0, 170, dt)
        age, sofa, C, g, parmsPD = fcnGeneratePatientParameters(N, "CV", 0.1)
        L0 = fcnGenerateStochasticTrajectories(t, parmsL, N)
        assert L0.shape == (N, t.size), f"L0 shape {L0.shape} != ({N}, {t.size})"

        # RCT sim
        T_rct = fcnSimulate_N_Patients(
            N, 1, np.full(N, 0.5), 0.1, C, g, 0.5, L0,
            np.array([10, 50]),
            np.array([-7, 0.3, 20, 5]),
            np.array([-5, 2.0, 0.1, -5, 2, 1.5]),
            age, sofa,
        )
        assert "sid" in T_rct.columns
        assert T_rct["sid"].nunique() == N
        print(f"  ok  RCT simulator (N={N}, rows={len(T_rct)})")

        # Dose-changing sim
        T_dc = fcnSimulate_DoseChanging(
            N, 0.1, C, g, 0.5, L0,
            np.array([-7, 0.3, 20, 5]),
            age, sofa, seed=0,
        )
        assert "sid" in T_dc.columns
        assert T_dc["sid"].nunique() == N
        print(f"  ok  Dose-changing simulator (N={N}, rows={len(T_dc)})")

    except Exception as e:
        failures += 1
        print(f"  FAIL tiny simulation: {e}")
        traceback.print_exc()

    return failures


def test_parity_harness() -> int:
    """Make sure verify_parity.py at least imports without error."""
    failures = 0
    try:
        sys.path.insert(0, str(HERE.parent))
        import verify_parity  # noqa: F401
        print(f"  ok  verify_parity import")
    except Exception as e:
        failures += 1
        print(f"  FAIL verify_parity import: {e}")
        traceback.print_exc()
    return failures


def main() -> int:
    total_failures = 0
    print("== test_imports ==")
    total_failures += test_imports()
    print("\n== test_tiny_simulation ==")
    total_failures += test_tiny_simulation()
    print("\n== test_parity_harness ==")
    total_failures += test_parity_harness()
    print()
    if total_failures == 0:
        print("ALL SMOKE TESTS PASSED")
        return 0
    else:
        print(f"{total_failures} SMOKE TEST FAILURE(S)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
