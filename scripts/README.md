# Reproducible Analytical & Visualization Pipeline (`scripts/`)

**Manuscript:** *Decarbonizing Turkish Black Sea Commercial Fisheries Requires Pairing Industrial Engine-Power Limits with Capacity Reduction*  
**Turkish Title (*Türkçe Başlık*):** *Türkiye Karadeniz Ticari Balıkçılığının Karbonsuzlaştırılması Endüstriyel Motor Gücü Sınırları ile Kapasite Azaltımının Birlikte Uygulanmasını Gerektirir*  
**Journal:** *Turkish Journal of Maritime and Marine Sciences* (TrJMMS)  
**Author:** Sezgin Tunca, Ph.D. (ORCID: `0000-0001-6840-1466`)

---

## Overview

This directory contains the ordered, self-contained Python pipeline (`00` through `08`) that reproduces every empirical metric, naval-architectural resistance decomposition, spatial autocorrelation statistic, non-parametric frontier efficiency score, conditional policy simulation, multi-basin econometric specification, and 600-DPI publication figure reported in the main manuscript and Supplementary Information (Sections S1–S7, Tables S1–S11, Figures S1–S2).

To execute the complete pipeline sequentially from start to finish:

```bash
python3 scripts/00_run_all_pipeline.py
```

---

## Ordered Pipeline Architecture

| Step | Script | Purpose & Mathematical Mapping | Key Outputs Generated |
| :--- | :--- | :--- | :--- |
| **`00`** | `00_run_all_pipeline.py` | **Master Orchestrator:** Executes Steps `01` through `08` sequentially and verifies zero non-zero exit codes across the entire analytical workflow. | Console execution log & verification of all `tables/` and `figures/` outputs |
| **`01`** | `01_compile_fleet_and_economic_data.py` | **Fleet Architecture & Cost Accounts:** Processes the active GFCM GSA 29 commercial vessel register ($N = 6,584$ Turkish vessels; $675,722.62\text{ kW}$; $87,363.60\text{ GT}$), cross-national Black Sea riparian capacity registers ($N = 9,105$ regional vessels; $N = 9,009$ motorized), and longitudinal TÜİK/BSGM operational cost accounts ($2000\text{--}2024$). Implements Eq. (1) ($\text{EII}_v = P_{\text{main},v} / \text{GT}_v$) and Section 2.4 operational cost accounting formulas. | `tables/turkish_fleet_segment_architecture.csv` (Table 2, Table S3)<br>`tables/blacksea_regional_fleet_comparison.csv` (Table S3b)<br>`tables/turkish_operational_cost_summary.csv` (Table S7) |
| **`02`** | `02_compile_fao_catch_benchmarks.py` | **Regional FAO Area 37 Catch & Longitudinal FUI:** Extracts historical marine capture landings ($1950\text{--}2024$; cumulative $41,861.36\text{ kt}$) across Black Sea riparian states, reconciles FAO Major Fishing Area 37 national totals against GSA 29 Black Sea-only landings ($256.74\text{ kt yr}^{-1}$), and compiles the Turkish Area 37 Fuel Use Intensity (FUI) index. | `tables/fao_blacksea_country_annual_pivot.csv`<br>`tables/fao_blacksea_country_summary.csv` (Table S4)<br>`tables/fao_blacksea_species_summary.csv`<br>`tables/turkish_fleet_fui_dynamics.csv` (Table S7) |
| **`03`** | `03_simulate_decarbonisation_scenarios.py` | **Duty-Cycle $\text{CO}_2$ Inventory, Policy Scenarios, Fuel Shocks, Monte Carlo & Panel Econometrics:** Computes the primary gear-differentiated operational duty-cycle emissions baseline ($366.98\text{ kt CO}_2\text{ yr}^{-1}$; $137.00\text{ million L}$ diesel) vs. the uniform industrial upper bound ($784.92\text{ kt CO}_2\text{ yr}^{-1}$; Eq. 8–9, Eq. S5); simulates conditional Policy Scenarios A–E and Sensitivity $\text{E}_{750}$ with exact 42-vessel overlap accounting ($11,695.23\text{ kW}$ shared excess); evaluates 2023 diesel price shock elasticities across $\delta \in [8\%, 20\%]$ capital consumption schedules (Section 2.4); runs $N = 10,000$ surviving-fleet Monte Carlo simulations contrasting $0\%$ catch rebound ($\text{B}_0, \text{E}_0$) against $+5\%$ crowding relief ($\text{B}, \text{E}, \text{E}_{750}$); and estimates multi-basin rolling ($N = 105$, Newey-West HAC 4 lags) and non-overlapping block ($N = 25$, HC1) regressions (Eq. 10). | `tables/gear_duty_cycle_emissions_baseline.csv` (Table S9a)<br>`tables/decarbonisation_scenarios_summary.csv` (Table 5)<br>`tables/fuel_sensitivity_analysis.csv` (Table S8)<br>`tables/depreciation_sensitivity_analysis.csv` (Table S7b)<br>`tables/depreciation_segment_weighted.csv` (Table S7b)<br>`tables/turkish_decarbonisation_scenarios.csv` (Table 5, Table S9)<br>`tables/regression_table_rev_cv.csv` (Table 6, Table S10a)<br>`tables/regression_table_catch_cv.csv` (Table S10b)<br>`tables/regression_non_overlapping_robustness.csv` (Table S10c) |
| **`04`** | `04_model_hydrodynamic_froude_scaling.py` | **Naval Architecture, Froude Regimes & Holtrop-Mennen Resistance:** Estimates within-gear log-log OLS and HC3 robust power-law scaling regressions of $\ln(\text{EII}_v)$ on $\ln(\text{LOA}_v)$ (Eq. 3); evaluates displacement-normalized propulsion intensity ($P_{\text{main},v} / \Delta_v$, $\text{kW t}^{-1}$); and computes Froude numbers (Eq. 2) plus calm-water viscous ($R_V$, ITTC-1957 + Holtrop 3D form factor + ITTC-1978 roughness allowance; Eq. 4, Eq. S1–S3) and wave-making ($R_W$, Holtrop & Mennen 1982, 1984 cross-validated against van Oortmerssen 1971; Eq. 5, Eq. S4) resistance shares across free-running transit and active working speeds. | `tables/froude_scaling_regressions_turkiye.csv` (Table 3, Table S5)<br>`tables/hydrodynamic_regime_summary_turkiye.csv` (Table S6) |
| **`05`** | `05_model_dea_eco_efficiency.py` | **Cohort Data Envelopment Analysis (DEA) & Directional Distance Frontiers:** Audits sample filtering from $N = 6,584$ registered vessels to $n = 4,772$ active East/West Black Sea vessels ($\ge 6\text{ m}$ LOA) across $K = 21$ homogeneous cohorts; solves output-oriented CCR ($\text{TE}_{\text{CRS}}$), BCC ($\text{TE}_{\text{VRS}}$), Andersen-Petersen super-efficiency, single-output ($s = 1$) dimensionality checks (Eq. 6), and Chung-Färe Directional Distance Function (DDF) eco-efficiency linear programs (Eq. 7); and tests sensitivity across four catch-allocation rules plus an expanded 25-cohort sample ($n = 6,563$). | `tables/dea_sample_exclusion_audit.csv` (Table S5b)<br>`tables/turkish_dea_gear_segment_results.csv` (Table 4)<br>`tables/turkish_dea_cohort_results.csv` (Table S6b)<br>`tables/dea_sensitivity_allocation_rules.csv` (Table S6c) |
| **`06`** | `06_analyze_spatial_autocorrelation.py` | **Port-Level Geocoding, Global Moran's $I$ & Dual LISA Clustering:** Geocodes all 34 registration ports ($27$ Black Sea/Bosphorus coastal ports [$n = 6,420$] + $7$ Sea of Marmara GSA 29 interface homeports [$n = 164$]); constructs row-standardized $k = 4$ (and $k = 3$ sensitivity) nearest-neighbor coastal spatial weight matrices; and computes permutation-based Global Moran's $I$ and Local Indicators of Spatial Association ($I_p$) with Benjamini-Hochberg FDR adjustments ($q_{\text{FDR}}$) for both port median EII ($\text{kW GT}^{-1}$) and total installed power ($\text{kW}$) (Section 2.3). | `tables/turkish_port_level_capacity_all34.csv` (Table S1)<br>`tables/turkish_port_level_capacity.csv`<br>`tables/morans_i_turkiye_results.csv` (Table S2a)<br>`tables/lisa_port_clusters_turkiye.csv` (Table S2b) |
| **`07`** | `07_extract_blacksea_ais_validation.py` | **Illustrative Satellite AIS Working-Speed Verification:** Processes 849 Global Fishing Watch (GFW) satellite AIS fishing-event records across 20 industrial vessels operating in the Black Sea ($2018\text{--}2024$), cross-references ITU Radio Call Signs (IRCS) and MMSI identifiers against GFCM GSA 29 registry records, and renders Supplementary Figure S2 at 600 DPI. | `tables/blacksea_ais_vessel_telemetry_validation.csv` (Table S11)<br>`figures/figS2_ais_speed_validation.png` & `.pdf` (Figure S2) |
| **`08`** | `08_generate_publication_figures.py` | **Master Publication Visualization Suite (600 DPI PNG & Vector PDF):** Renders all main-text figures (Figures 1–7) and Supplementary Figure S1 directly from the verified analytical tables and vessel/port outputs. | `figures/fig1_methodological_workflow.png` & `.pdf`<br>`figures/fig2_study_area_map.png` & `.pdf`<br>`figures/fig3_hydrodynamic_froude_scaling.png` & `.pdf`<br>`figures/fig4_comparative_blacksea_dynamics.png` & `.pdf`<br>`figures/fig5_turkish_port_bubble_maps.png` & `.pdf`<br>`figures/fig6_decarbonisation_tradeoffs.png` & `.pdf`<br>`figures/fig7_dea_eco_efficiency_frontiers.png` & `.pdf`<br>`figures/figS1_supplementary_diagnostics.png` & `.pdf` |

---

## Software Environment & Dependencies

- **Python:** $\ge 3.10$
- **Core Scientific Stack:**
  - `numpy >= 1.24.0`
  - `pandas >= 2.0.0`
  - `scipy >= 1.11.0` (`scipy.optimize.linprog` HiGHS solver)
  - `statsmodels >= 0.14.0` (OLS, HC1/HC3 robust covariance, Newey-West HAC)
  - `geopandas >= 0.14.0`, `shapely >= 2.0.0`, `libpysal >= 4.9.0`, `esda >= 2.5.0` (spatial weights, Global Moran's $I$, LISA)
  - `matplotlib >= 3.7.0`, `seaborn >= 0.13.0` (600-DPI visualization suite)

Install all dependencies via:

```bash
pip install -r requirements.txt
```
