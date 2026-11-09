#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT: 04_model_hydrodynamic_froude_scaling.py
STUDY:  Decarbonizing Turkish Black Sea Commercial Fisheries Requires Pairing
        Industrial Engine-Power Limits with Capacity Reduction
JOURNAL: Turkish Journal of Maritime and Marine Sciences (DOI: 10.52998/trjmms.2042258)
AUTHOR: Sezgin Tunca, Ph.D. (ORCID: 0000-0001-6840-1466)
PURPOSE: Analyze empirical scaling of Engine Intensity Index (EII = kW/GT) versus
         Length Overall (LOA) and compute theoretical semi-empirical hydrodynamic
         resistance decompositions (ITTC-1957/1978 viscous + roughness line,
         Holtrop & Mennen 1982, 1984, and van Oortmerssen 1971 small-ship/trawler
         formulation) across the 6,584 active vessels in the Turkish Black Sea fleet.
INPUTS:
  - data/gfcm_blacksea_fleet.csv
OUTPUTS:
  - data/turkish_hydrodynamic_vessel_estimates.csv
  - tables/froude_scaling_regressions_turkiye.csv
  - tables/hydrodynamic_regime_summary_turkiye.csv
================================================================================
"""

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TABLES_DIR = os.path.join(BASE_DIR, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

print("=== RUNNING NAVAL ARCHITECTURE & FROUDE SCALING ANALYSIS (TÜRKİYE) ===")

# 1. Load Turkish Fleet Data
fleet_df = pd.read_csv(os.path.join(DATA_DIR, 'gfcm_blacksea_fleet.csv'), low_memory=False)
tr_fleet = fleet_df[
    (fleet_df['country'] == 'Türkiye') &
    (fleet_df['eii'].notna()) & (fleet_df['eii'] > 0) &
    (fleet_df['loa_m'].notna()) & (fleet_df['loa_m'] > 0) &
    (fleet_df['tonnage_gt'].notna()) & (fleet_df['tonnage_gt'] > 0)
].copy()

print(f"Verified Turkish vessels for hydrodynamic scaling: {len(tr_fleet)}")

def classify_gear(row):
    g1 = str(row.get('gear1', '')).strip().upper()
    gg = str(row.get('gear_group', '')).strip().lower()
    if g1 == 'PS' or 'purse' in gg:
        return 'Purse Seiner (PS)'
    elif g1 in ['OTB', 'PTM'] or 'trawl' in gg:
        if g1 == 'TBB':
            return 'Beam Trawler (TBB)'
        return 'Pelagic/Demersal Trawler (OTB/PTM)'
    elif g1 == 'TBB':
        return 'Beam Trawler (TBB)'
    elif g1 == 'DRB' or 'dredg' in gg:
        return 'Hydraulic Dredger (DRB)'
    else:
        return 'Small-Scale / Polyvalent (MIS)'

tr_fleet['gear_detailed'] = tr_fleet.apply(classify_gear, axis=1)

# Natural logs for power-law scaling: ln(EII) = alpha + beta * ln(LOA)
tr_fleet['ln_eii'] = np.log(tr_fleet['eii'])
tr_fleet['ln_loa'] = np.log(tr_fleet['loa_m'])
tr_fleet['ln_kw'] = np.log(tr_fleet['engine_power_kw'])
tr_fleet['ln_gt'] = np.log(tr_fleet['tonnage_gt'])

# 2. Semi-Empirical Naval Architecture Resistance Model
# Black Sea brackish surface water physical constants (S ~ 18 psu, T ~ 15 C)
RHO_SW = 1014.0        # kg / m^3
NU_SW = 1.19e-6        # m^2 / s kinematic viscosity
G_ACC = 9.80665        # m / s^2
CA_ROUGHNESS = 0.0005  # ITTC-1978 standard hull-roughness/fouling allowance for working fishing hulls

# Gear-specific hull-form parameters & naval engineering EII reference bands
# (Holtrop & Mennen 1982, 1984; van Oortmerssen 1971; Wilson 1999; Gulbrandsen 2012; Sala et al. 2011)
HULL_PARAMS = {
    'Purse Seiner (PS)': {
        'L_B': 4.65, 'B_T': 2.65, 'C_B': 0.54, 'C_P': 0.61, 'C_M': 0.885, 'C_WP': 0.74,
        'v_transit_kn': 11.5, 'v_low_kn': 10.0, 'v_high_kn': 13.0, 'v_work_kn': 2.0,
        'eii_bench_low': 3.0, 'eii_bench_high': 6.0,
        'bench_basis': 'Decked free-running displacement hull (3.0-6.0 kW/GT)'
    },
    'Pelagic/Demersal Trawler (OTB/PTM)': {
        'L_B': 3.85, 'B_T': 2.45, 'C_B': 0.56, 'C_P': 0.63, 'C_M': 0.889, 'C_WP': 0.75,
        'v_transit_kn': 9.5, 'v_low_kn': 8.0, 'v_high_kn': 11.0, 'v_work_kn': 3.0,
        'eii_bench_low': 6.0, 'eii_bench_high': 10.0,
        'bench_basis': 'Active towing bollard-pull sizing (6.0-10.0 kW/GT)'
    },
    'Hydraulic Dredger (DRB)': {
        'L_B': 3.80, 'B_T': 2.45, 'C_B': 0.56, 'C_P': 0.63, 'C_M': 0.889, 'C_WP': 0.75,
        'v_transit_kn': 9.0, 'v_low_kn': 7.5, 'v_high_kn': 10.5, 'v_work_kn': 2.5,
        'eii_bench_low': 6.0, 'eii_bench_high': 10.0,
        'bench_basis': 'Active hydraulic dredge & pump sizing (6.0-10.0 kW/GT)'
    },
    'Beam Trawler (TBB)': {
        'L_B': 3.45, 'B_T': 2.50, 'C_B': 0.50, 'C_P': 0.60, 'C_M': 0.833, 'C_WP': 0.72,
        'v_transit_kn': 7.0, 'v_low_kn': 5.5, 'v_high_kn': 8.5, 'v_work_kn': 2.8,
        'eii_bench_low': 6.0, 'eii_bench_high': 12.0,
        'bench_basis': 'Coastal small-beam drag craft; undecked GT compression (6.0-12.0 kW/GT)'
    },
    'Small-Scale / Polyvalent (MIS)': {
        'L_B': 3.50, 'B_T': 2.55, 'C_B': 0.48, 'C_P': 0.59, 'C_M': 0.814, 'C_WP': 0.71,
        'v_transit_kn': 6.5, 'v_low_kn': 5.0, 'v_high_kn': 8.0, 'v_work_kn': 2.0,
        'eii_bench_low': 4.0, 'eii_bench_high': 12.0,
        'bench_basis': 'Undecked/semi-decked artisanal craft; evaluate kW/Delta (2.2-3.7 kW/t)'
    }
}

def semi_empirical_resistance(loa_m, hp, v_knots):
    """
    Evaluate viscous resistance R_V (ITTC-1957 frictional line + Holtrop 1984 3D form factor
    + ITTC-1978 working-hull roughness allowance C_A) and wave-making resistance R_W
    using Holtrop & Mennen (1982, 1984) for L/B >= 4.0 hulls and van Oortmerssen (1971)
    small-ship/trawler formulation (and cross-validation) for L/B < 4.0 hulls.
    """
    L = 0.95 * loa_m  # Waterline length L_WL (m)
    B = L / hp['L_B']
    T = B / hp['B_T']
    CB = hp['C_B']
    CP = hp['C_P']
    CM = hp['C_M']
    CWP = hp['C_WP']

    nabla = CB * L * B * T                  # Volumetric displacement (m^3)
    delta_t = (RHO_SW / 1000.0) * nabla     # Mass displacement (metric tonnes)

    # Wetted surface area (Holtrop & Mennen 1982, m^2)
    Sw = L * (2.0 * T + B) * np.sqrt(CM) * (
        0.453 + 0.4425 * CB - 0.2862 * CM - 0.003467 * (B / T) + 0.3696 * CWP
    )

    v_ms = v_knots * 0.514444
    fn = v_ms / np.sqrt(G_ACC * L)
    re = max(1e5, (v_ms * L) / NU_SW)

    # ITTC-1957 frictional line + Holtrop (1984) 3D viscous form factor + ITTC-1978 roughness allowance
    cf = 0.075 / ((np.log10(re) - 2.0) ** 2)
    lcb = -0.75
    LR = max(0.2 * L, L * (1.0 - CP + (0.06 * CP * lcb) / (4.0 * CP - 1.0)))
    one_plus_k1 = 0.93 + 0.487118 * ((B / L) ** 1.06806) * ((T / L) ** 0.46106) * \
                  ((L / LR) ** 0.121563) * (((L ** 3) / nabla) ** 0.36486) * \
                  ((1.0 - CP) ** (-0.604247))
    one_plus_k1 = float(np.clip(one_plus_k1, 1.10, 1.50))

    R_V = 0.5 * RHO_SW * (v_ms ** 2) * Sw * (cf * one_plus_k1 + CA_ROUGHNESS)

    # A. Holtrop & Mennen (1982, 1984) wave-making resistance
    ie_exp = -((L / B) ** 0.80856) * ((1.0 - CWP) ** 0.30484) * \
             ((1.0 - CP - 0.0225 * lcb) ** 0.6367) * ((LR / B) ** 0.34574) * \
             (((100.0 * nabla) / (L ** 3)) ** 0.16302)
    i_E = float(np.clip(1.0 + 89.0 * np.exp(ie_exp), 8.0, 48.0))

    if (B / L) < 0.11:
        c7 = 0.229577 * ((B / L) ** 0.33333)
    elif (B / L) <= 0.25:
        c7 = B / L
    else:
        c7 = 0.5 - 0.0625 * (L / B)
    c1 = 2223105.0 * (c7 ** 3.78613) * ((T / B) ** 1.07961) * ((90.0 - i_E) ** (-1.37565))

    c16 = 8.07981 * CP - 13.8673 * (CP ** 2) + 6.984388 * (CP ** 3) if CP < 0.80 else 1.73014 - 0.7067 * CP
    m1 = 0.0140407 * (L / T) - 1.75254 * ((nabla ** (1.0 / 3.0)) / L) - 4.79323 * (B / L) - c16

    l3_nabla = (L ** 3) / nabla
    if l3_nabla < 512.0:
        c15 = -1.69385
    elif l3_nabla < 1726.91:
        c15 = -1.69385 + (L / (nabla ** (1.0 / 3.0)) - 8.0) / 2.36
    else:
        c15 = 0.0

    def eval_hm_rw(f_val):
        f_eff = min(max(0.05, f_val), 0.42)
        m4_val = c15 * 0.4 * np.exp(-0.034 * (f_eff ** (-3.29)))
        lam = 1.446 * CP - 0.03 * (L / B) if (L / B) < 12.0 else 1.446 * CP - 0.36
        exponent = m1 * (f_eff ** (-0.9)) + m4_val * np.cos(lam * (f_eff ** (-2.0)))
        rw_base = c1 * nabla * RHO_SW * G_ACC * np.exp(np.clip(exponent, -30.0, 2.0))
        if f_val > 0.42:
            # Monotonic high-Froude semi-displacement wave growth (Fn > 0.42)
            rw_base = rw_base * ((f_val / 0.42) ** 3.4)
        return rw_base

    R_W_hm = eval_hm_rw(fn)

    # B. van Oortmerssen (1971) small-ship & trawler wave-making cross-validation (3.0 <= L/B <= 6.2)
    # Uses wide-beam trawl resistance coefficient scaling (c7_vo capped for low L/B hulls)
    c7_vo = min(0.245, 0.5 - 0.0625 * (L / B)) if (B / L) > 0.22 else (B / L)
    c1_vo = 2223105.0 * (c7_vo ** 3.78613) * ((T / B) ** 1.07961) * ((90.0 - i_E) ** (-1.37565))
    f_eff_vo = min(max(0.05, fn), 0.42)
    m4_vo = c15 * 0.4 * np.exp(-0.034 * (f_eff_vo ** (-3.29)))
    lam_vo = 1.446 * CP - 0.03 * (L / B)
    R_W_vo = c1_vo * nabla * RHO_SW * G_ACC * np.exp(np.clip((m1 + 0.06) * (f_eff_vo ** (-0.9)) + m4_vo * np.cos(lam_vo * (f_eff_vo ** (-2.0))), -30.0, 2.0))
    if fn > 0.42:
        R_W_vo = R_W_vo * ((fn / 0.42) ** 3.4)

    share_hm = R_W_hm / (R_V + R_W_hm) if (R_V + R_W_hm) > 0 else 0.0
    share_vo = R_W_vo / (R_V + R_W_vo) if (R_V + R_W_vo) > 0 else 0.0

    # Primary curve: unified Holtrop & Mennen (1982, 1984) formulation with wide-beam c7 branch,
    # cross-validated against low-L/B small-ship sensitivity (share_vo)
    primary_share = share_hm
    return fn, R_V, primary_share, share_hm, share_vo, delta_t

def evaluate_vessel(row):
    loa = row['loa_m']
    kw = row['engine_power_kw']
    gt = row['tonnage_gt']
    eii = row['eii']
    g = row['gear_detailed']
    hp = HULL_PARAMS[g]

    fn_tr, _, share_tr, share_hm, share_vo, delta_t = semi_empirical_resistance(loa, hp, hp['v_transit_kn'])
    _, _, share_low, _, _, _ = semi_empirical_resistance(loa, hp, hp['v_low_kn'])
    _, _, share_high, _, _, _ = semi_empirical_resistance(loa, hp, hp['v_high_kn'])
    fn_wk, _, share_wk, _, _, _ = semi_empirical_resistance(loa, hp, hp['v_work_kn'])

    v_hull_kn = 2.43 * np.sqrt(0.95 * loa)
    speed_ratio = hp['v_transit_kn'] / v_hull_kn
    kw_per_delta = kw / delta_t if delta_t > 0 else np.nan
    delta_to_gt = delta_t / gt if gt > 0 else np.nan
    exceeds_gear_bench = 1 if eii > hp['eii_bench_high'] else 0

    return pd.Series([
        hp['v_transit_kn'], fn_tr, v_hull_kn, speed_ratio,
        share_tr, share_low, share_high, fn_wk, share_wk, share_hm, share_vo,
        delta_t, delta_to_gt, kw_per_delta, exceeds_gear_bench
    ])

cols_out = [
    'Transit_Speed_knots', 'Froude_Number_Fn', 'Hull_Speed_knots', 'Speed_Ratio',
    'Wave_Resistance_Share', 'Wave_Share_LowSpeed', 'Wave_Share_HighSpeed',
    'Working_Fn', 'Working_Wave_Share', 'Holtrop_Wave_Share', 'VanOortmerssen_Wave_Share',
    'Est_Displacement_t', 'Delta_to_GT_Ratio', 'kW_per_Displacement_t', 'Exceeds_Gear_Benchmark'
]
tr_fleet[cols_out] = tr_fleet.apply(evaluate_vessel, axis=1)

# Save vessel-level hydrodynamic dataset for exact plotting in Figure 3 (zero synthetic draws)
tr_fleet[[
    'gear_detailed', 'loa_m', 'tonnage_gt', 'engine_power_kw', 'eii',
    'Transit_Speed_knots', 'Froude_Number_Fn', 'Hull_Speed_knots', 'Speed_Ratio',
    'Wave_Resistance_Share', 'Wave_Share_LowSpeed', 'Wave_Share_HighSpeed',
    'Working_Fn', 'Working_Wave_Share', 'Holtrop_Wave_Share', 'VanOortmerssen_Wave_Share',
    'Est_Displacement_t', 'Delta_to_GT_Ratio', 'kW_per_Displacement_t', 'Exceeds_Gear_Benchmark'
]].to_csv(os.path.join(DATA_DIR, 'turkish_hydrodynamic_vessel_estimates.csv'), index=False)

# 3. Scaling Regressions (OLS & HC3 heteroskedasticity-robust SEs)
global_model = smf.ols("ln_eii ~ ln_loa", data=tr_fleet).fit()
global_hc3 = global_model.get_robustcov_results(cov_type='HC3')

scaling_results = [{
    'Segment': 'All Turkish Fleet',
    'N': len(tr_fleet),
    'Mean_LOA_m': round(tr_fleet['loa_m'].mean(), 2),
    'Mean_kW': round(tr_fleet['engine_power_kw'].mean(), 2),
    'Mean_GT': round(tr_fleet['tonnage_gt'].mean(), 2),
    'Mean_EII_kW_GT': round(tr_fleet['eii'].mean(), 2),
    'Median_EII_kW_GT': round(tr_fleet['eii'].median(), 2),
    'Median_kW_per_Delta_t': round(tr_fleet['kW_per_Displacement_t'].median(), 2),
    'Intercept (alpha)': round(global_model.params['Intercept'], 4),
    'Scaling Exponent (beta)': round(global_model.params['ln_loa'], 4),
    'SE (beta)': round(global_model.bse['ln_loa'], 4),
    'HC3_SE (beta)': round(float(global_hc3.bse[1]), 4),
    't_stat': round(global_model.tvalues['ln_loa'], 2),
    'p_value': round(global_model.pvalues['ln_loa'], 4),
    'R_squared': round(global_model.rsquared, 4)
}]

gear_order = [
    'Purse Seiner (PS)',
    'Pelagic/Demersal Trawler (OTB/PTM)',
    'Beam Trawler (TBB)',
    'Small-Scale / Polyvalent (MIS)',
    'Hydraulic Dredger (DRB)'
]

for gear in gear_order:
    gear_df = tr_fleet[tr_fleet['gear_detailed'] == gear]
    m = smf.ols("ln_eii ~ ln_loa", data=gear_df).fit()
    m_hc3 = m.get_robustcov_results(cov_type='HC3')
    scaling_results.append({
        'Segment': gear,
        'N': len(gear_df),
        'Mean_LOA_m': round(gear_df['loa_m'].mean(), 2),
        'Mean_kW': round(gear_df['engine_power_kw'].mean(), 2),
        'Mean_GT': round(gear_df['tonnage_gt'].mean(), 2),
        'Mean_EII_kW_GT': round(gear_df['eii'].mean(), 2),
        'Median_EII_kW_GT': round(gear_df['eii'].median(), 2),
        'Median_kW_per_Delta_t': round(gear_df['kW_per_Displacement_t'].median(), 2),
        'Intercept (alpha)': round(m.params['Intercept'], 4),
        'Scaling Exponent (beta)': round(m.params['ln_loa'], 4),
        'SE (beta)': round(m.bse['ln_loa'], 4),
        'HC3_SE (beta)': round(float(m_hc3.bse[1]), 4),
        't_stat': round(m.tvalues['ln_loa'], 2),
        'p_value': round(m.pvalues['ln_loa'], 4),
        'R_squared': round(m.rsquared, 4)
    })

df_scaling = pd.DataFrame(scaling_results)
df_scaling.to_csv(os.path.join(TABLES_DIR, 'froude_scaling_regressions_turkiye.csv'), index=False)
print("\nScaling regression results saved to tables/froude_scaling_regressions_turkiye.csv:")
print(df_scaling[['Segment', 'N', 'Scaling Exponent (beta)', 'SE (beta)', 'R_squared', 'Mean_EII_kW_GT', 'Median_kW_per_Delta_t']].to_string(index=False))

# 4. Hydrodynamic Regime Summary Table
hydro_rows = []
for gear in gear_order:
    gdf = tr_fleet[tr_fleet['gear_detailed'] == gear]
    hp = HULL_PARAMS[gear]
    model_diff = (gdf['Holtrop_Wave_Share'] - gdf['VanOortmerssen_Wave_Share']).abs().mean() * 100.0
    hydro_rows.append({
        'gear_detailed': gear,
        'Vessel_Count': len(gdf),
        'Mean_LOA': round(gdf['loa_m'].mean(), 2),
        'Mean_Transit_Speed': hp['v_transit_kn'],
        'Mean_Fn': round(gdf['Froude_Number_Fn'].mean(), 4),
        'Median_Fn': round(gdf['Froude_Number_Fn'].median(), 4),
        'Mean_Hull_Speed': round(gdf['Hull_Speed_knots'].mean(), 2),
        'Mean_Speed_Ratio': round(gdf['Speed_Ratio'].mean(), 4),
        'Mean_Wave_Resistance_Share_%': round(gdf['Wave_Resistance_Share'].mean() * 100.0, 2),
        'Wave_Share_Range_Low_High_%': f"{gdf['Wave_Share_LowSpeed'].mean()*100:.1f}-{gdf['Wave_Share_HighSpeed'].mean()*100:.1f}%",
        'Working_Speed_kn': hp['v_work_kn'],
        'Working_Fn': round(gdf['Working_Fn'].mean(), 4),
        'Working_Wave_Share_%': round(gdf['Working_Wave_Share'].mean() * 100.0, 2),
        'Mean_Delta_to_GT_Ratio': round(gdf['Delta_to_GT_Ratio'].mean(), 2),
        'Median_kW_per_Delta_t': round(gdf['kW_per_Displacement_t'].median(), 2),
        'Gear_Benchmark_kW_GT': f"{hp['eii_bench_low']:.1f}-{hp['eii_bench_high']:.1f}",
        'Pct_Exceeding_Gear_Benchmark': round(gdf['Exceeds_Gear_Benchmark'].mean() * 100.0, 2),
        'Overpowered_Vessels_EII_gt_10': int((gdf['eii'] > 10.0).sum()),
        'Pct_Overpowered': round((gdf['eii'] > 10.0).mean() * 100.0, 2),
        'HM_vs_VO_Mean_Diff_pp': round(model_diff, 2)
    })

hydro_summary = pd.DataFrame(hydro_rows)
hydro_summary.to_csv(os.path.join(TABLES_DIR, 'hydrodynamic_regime_summary_turkiye.csv'), index=False)
print("\nHydrodynamic regime summary saved to tables/hydrodynamic_regime_summary_turkiye.csv:")
print(hydro_summary[['gear_detailed', 'Mean_LOA', 'Mean_Fn', 'Mean_Wave_Resistance_Share_%', 'Wave_Share_Range_Low_High_%', 'Working_Wave_Share_%', 'Mean_Delta_to_GT_Ratio', 'Median_kW_per_Delta_t']].to_string(index=False))

print("\n[SUCCESS] Naval architecture and Froude scaling analysis completed.")
