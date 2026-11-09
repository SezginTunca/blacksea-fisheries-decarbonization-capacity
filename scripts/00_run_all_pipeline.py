#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT: 00_run_all_pipeline.py
STUDY:  Decarbonizing Turkish Black Sea Commercial Fisheries Requires Pairing
        Industrial Engine-Power Limits with Capacity Reduction
JOURNAL: Turkish Journal of Maritime and Marine Sciences (DOI: 10.52998/trjmms.2042258)
AUTHOR: Sezgin Tunca, Ph.D. (ORCID: 0000-0001-6840-1466)
PURPOSE: Master analytical pipeline orchestrator. Executes Steps 01 through 08
         sequentially to reproduce all empirical tables, naval architectural
         decompositions, frontier efficiency scores, spatial autocorrelation
         statistics, econometric models, and 600-DPI publication figures.

Usage:
    python3 scripts/00_run_all_pipeline.py
================================================================================
"""

import os
import subprocess
import sys
import time

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    "01_compile_fleet_and_economic_data.py",
    "02_compile_fao_catch_benchmarks.py",
    "03_simulate_decarbonisation_scenarios.py",
    "04_model_hydrodynamic_froude_scaling.py",
    "05_model_dea_eco_efficiency.py",
    "06_analyze_spatial_autocorrelation.py",
    "07_extract_blacksea_ais_validation.py",
    "08_generate_publication_figures.py",
]


def main():
    t_all = time.time()
    for step in STEPS:
        print("\n" + "=" * 80 + f"\nRUNNING {step}\n" + "=" * 80, flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, os.path.join(SCRIPTS_DIR, step)])
        print(f"--> {step} finished in {time.time() - t0:.1f} s (exit code {rc})", flush=True)
        if rc != 0:
            print(f"[FAILED] Pipeline aborted at {step}.")
            sys.exit(rc)
    print(f"\n[SUCCESS] All {len(STEPS)} pipeline steps completed in {time.time() - t_all:.1f} s.")


if __name__ == "__main__":
    main()
