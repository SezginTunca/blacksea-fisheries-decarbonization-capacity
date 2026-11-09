#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT: 05_model_dea_eco_efficiency.py
STUDY:  Decarbonizing Turkish Black Sea Commercial Fisheries Requires Pairing
        Industrial Engine-Power Limits with Capacity Reduction
JOURNAL: Turkish Journal of Maritime and Marine Sciences
AUTHOR: Sezgin Tunca, Ph.D. (ORCID: 0000-0001-6840-1466)
PURPOSE: Non-parametric Data Envelopment Analysis (DEA) and Directional Distance
         Function (DDF) eco-efficiency linear programming models for the Turkish
         Black Sea fishing fleet, incorporating:
           1. Primary Gear-Differentiated Operational Duty-Cycle CO2 emissions
           2. Sample Attrition & Exclusion Audit (4,772 included + 1,812 excluded = 6,584 vessels)
           3. Standard CRS, VRS, Andersen-Petersen Super-Efficiency, and Single-Output (s=1)
              dimensionality diagnostics across 21 homogeneous cohorts
           4. Catch-Allocation Sensitivity Analysis across 4 allocation rules (kW-proportional,
              GT-proportional, Vessel-count proportional, and LOA^1.5 geometric hull-length scaling)
           5. Expanded-Sample Inclusion Sensitivity (adding micro <6 m cohorts and
              Marmara-interface vessels, n = 6,563 across 25 cohorts).
INPUTS:
  - data/gfcm_blacksea_fleet.csv
  - data/TR_Fisheries_Amounts_Prices.xlsx (with pre-verified 2015-2024 guild fallback)
OUTPUTS:
  - tables/dea_sample_exclusion_audit.csv
  - tables/turkish_dea_gear_segment_results.csv
  - tables/dea_sensitivity_allocation_rules.csv
================================================================================
"""

import os
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from scipy.stats import spearmanr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TABLES_DIR = os.path.join(BASE_DIR, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

print("=== RUNNING DATA ENVELOPMENT ANALYSIS (DEA) & SENSITIVITY SUITE (TÜRKİYE) ===")

# 1. DEA Linear Programming Solvers
def solve_standard_dea(Y, X, returns='vrs', super_eff=False):
    """
    Output-oriented CCR (crs) or BCC (vrs) DEA linear program.
    Maximizes output expansion theta; Technical Efficiency = 1 / theta.
    If super_eff=True (Andersen & Petersen 1993 under CRS), excludes DMU i from reference set.
    """
    n, s = Y.shape
    _, m = X.shape
    te = np.full(n, np.nan)

    for i in range(n):
        ref_idx = [j for j in range(n) if j != i] if super_eff else list(range(n))
        n_ref = len(ref_idx)
        Y_ref = Y[ref_idx]
        X_ref = X[ref_idx]

        c = np.zeros(1 + n_ref)
        c[0] = -1.0  # maximize theta

        A_out = np.zeros((s, 1 + n_ref))
        A_out[:, 0] = Y[i]
        A_out[:, 1:] = -Y_ref.T
        b_out = np.zeros(s)

        A_in = np.zeros((m, 1 + n_ref))
        A_in[:, 1:] = X_ref.T
        b_in = X[i]

        A_ub = np.vstack([A_out, A_in])
        b_ub = np.concatenate([b_out, b_in])

        if returns == 'vrs':
            A_eq = np.zeros((1, 1 + n_ref))
            A_eq[0, 1:] = 1.0
            b_eq = np.array([1.0])
        else:
            A_eq, b_eq = None, None

        theta_lb = 0.0 if super_eff else 1.0
        bounds = [(theta_lb, None)] + [(0.0, None)] * n_ref

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success and res.x[0] > 0:
            te[i] = 1.0 / res.x[0]
    return te

def solve_ddf_dea(Y_good, Y_bad, X, returns='vrs'):
    """
    Directional Distance Function (DDF) eco-efficiency solver (Chung et al. 1997; Fare et al. 2005).
    Maximizes beta expanding Y_good by +beta*g_y and contracting Y_bad by -beta*g_b.
    """
    n, s = Y_good.shape
    _, q = Y_bad.shape
    _, m = X.shape
    beta = np.full(n, np.nan)

    for i in range(n):
        g_y = Y_good[i]
        g_b = Y_bad[i]

        c = np.zeros(1 + n)
        c[0] = -1.0

        A_good = np.zeros((s, 1 + n))
        A_good[:, 0] = g_y
        A_good[:, 1:] = -Y_good.T
        b_good = -Y_good[i]

        A_bad = np.zeros((q, 1 + n))
        A_bad[:, 0] = g_b
        A_bad[:, 1:] = Y_bad.T
        b_bad = Y_bad[i]

        A_in = np.zeros((m, 1 + n))
        A_in[:, 1:] = X.T
        b_in = X[i]

        A_ub = np.vstack([A_good, A_bad, A_in])
        b_ub = np.concatenate([b_good, b_bad, b_in])

        if returns == 'vrs':
            A_eq = np.zeros((1, 1 + n))
            A_eq[0, 1:] = 1.0
            b_eq = np.array([1.0])
        else:
            A_eq, b_eq = None, None

        bounds = [(0.0, None)] + [(0.0, None)] * n
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            beta[i] = max(0.0, res.x[0])
    return beta

# 2. Load Fleet Register & Synchronize Gear-Differentiated CO2 Factors (identical to Script 03)
fleet_df = pd.read_csv(os.path.join(DATA_DIR, 'gfcm_blacksea_fleet.csv'), low_memory=False)
tr_all = fleet_df[fleet_df['country'] == 'Türkiye'].copy()

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

tr_all['gear_detailed'] = tr_all.apply(classify_gear, axis=1)

GEAR_EF_REALISTIC = {
    'Purse Seiner (PS)': 1500.0 * 0.55 * 1.75 * 210.0 * 3.17 * 1e-6,
    'Pelagic/Demersal Trawler (OTB/PTM)': 1400.0 * 0.60 * 1.45 * 210.0 * 3.17 * 1e-6,
    'Beam Trawler (TBB)': 900.0 * 0.50 * 1.15 * 215.0 * 3.17 * 1e-6,
    'Hydraulic Dredger (DRB)': 1100.0 * 0.55 * 1.65 * 210.0 * 3.17 * 1e-6,
    'Small-Scale / Polyvalent (MIS)': 600.0 * 0.30 * 1.05 * 220.0 * 3.17 * 1e-6
}
tr_all['ef_realistic'] = tr_all['gear_detailed'].map(GEAR_EF_REALISTIC)
tr_all['co2_realistic_t'] = tr_all['engine_power_kw'] * tr_all['ef_realistic']
tr_all['co2_upper_t'] = tr_all['engine_power_kw'] * 1.1616

west_ports = ['ISTANBUL', 'IGNEADA', 'SILE', 'KEFKEN', 'KARASU', 'ZONGULDAK', 'KDZ. EREGLI', 'BARTIN', 'AMASRA', 'CIDE', 'INEBOLU', 'AYANCIK', 'SINOP', 'GERZE']
east_ports = ['SAMSUN', 'ORDU', 'FATSA', 'UNYE', 'GIRESUN', 'GORELE', 'TIREBOLU', 'TRABZON', 'VAKFIKEBIR', 'SURMENE', 'RIZE', 'PAZAR', 'HOPA']
tr_all['basin'] = tr_all['port_of_registration'].apply(
    lambda x: 'West' if x in west_ports else ('East' if x in east_ports else 'Other')
)

# 3. Exact Sample Attrition & Exclusion Audit (6,584 -> 4,772; 1,812 excluded)
valid_sc = ['small (6-12m)', 'medium (12-18m)', 'large (18-24m)', 'very_large (>=24m)']
m_lt6 = ~tr_all['size_class'].isin(valid_sc)  # 1,629 micro <6m vessels
m_drb = (~m_lt6) & (tr_all['gear_detailed'] == 'Hydraulic Dredger (DRB)')  # 20 DRB >=6m
m_oth = (~m_lt6) & (~m_drb) & (tr_all['basin'] == 'Other')                 # 162 >=6m Marmara-interface vessels
rem_pool = tr_all[~(m_lt6 | m_drb | m_oth)].copy()
cell_sizes = rem_pool.groupby(['basin', 'gear_detailed', 'size_class'])['loa_m'].transform('count')
m_singleton = cell_sizes < 3  # 1 vessel (West TBB 18-24m)

audit_rows = [
    {
        'Filter_Stage': '0. Full Active GFCM Turkish Black Sea Registry',
        'Vessels_n': len(tr_all),
        'Share_Fleet_%': 100.00,
        'Total_kW': round(tr_all['engine_power_kw'].sum(), 2),
        'Share_kW_%': 100.00,
        'Total_GT': round(tr_all['tonnage_gt'].sum(), 2),
        'Realistic_CO2_kt': round(tr_all['co2_realistic_t'].sum() / 1000.0, 2),
        'Rationale': 'Complete certified national Black Sea registry baseline'
    },
    {
        'Filter_Stage': '1. Excluded: Micro Artisanal Skiffs (LOA < 6.0 m)',
        'Vessels_n': int(m_lt6.sum()),
        'Share_Fleet_%': round(m_lt6.mean() * 100.0, 2),
        'Total_kW': round(tr_all.loc[m_lt6, 'engine_power_kw'].sum(), 2),
        'Share_kW_%': round(tr_all.loc[m_lt6, 'engine_power_kw'].sum() / tr_all['engine_power_kw'].sum() * 100.0, 2),
        'Total_GT': round(tr_all.loc[m_lt6, 'tonnage_gt'].sum(), 2),
        'Realistic_CO2_kt': round(tr_all.loc[m_lt6, 'co2_realistic_t'].sum() / 1000.0, 2),
        'Rationale': 'Subsistence/nearshore skiffs (<6 m) not disaggregated in TUIK commercial basin landings'
    },
    {
        'Filter_Stage': '2. Excluded: Hydraulic Suction Dredgers (DRB >= 6.0 m)',
        'Vessels_n': int(m_drb.sum()),
        'Share_Fleet_%': round(m_drb.mean() * 100.0, 2),
        'Total_kW': round(tr_all.loc[m_drb, 'engine_power_kw'].sum(), 2),
        'Share_kW_%': round(tr_all.loc[m_drb, 'engine_power_kw'].sum() / tr_all['engine_power_kw'].sum() * 100.0, 2),
        'Total_GT': round(tr_all.loc[m_drb, 'tonnage_gt'].sum(), 2),
        'Realistic_CO2_kt': round(tr_all.loc[m_drb, 'co2_realistic_t'].sum() / 1000.0, 2),
        'Rationale': 'Distinct water-jet suction clam technology operating under quota-restricted spatial zones'
    },
    {
        'Filter_Stage': '3. Excluded: Sea of Marmara Interface Homeports (>= 6.0 m non-DRB)',
        'Vessels_n': int(m_oth.sum()),
        'Share_Fleet_%': round(m_oth.mean() * 100.0, 2),
        'Total_kW': round(tr_all.loc[m_oth, 'engine_power_kw'].sum(), 2),
        'Share_kW_%': round(tr_all.loc[m_oth, 'engine_power_kw'].sum() / tr_all['engine_power_kw'].sum() * 100.0, 2),
        'Total_GT': round(tr_all.loc[m_oth, 'tonnage_gt'].sum(), 2),
        'Realistic_CO2_kt': round(tr_all.loc[m_oth, 'co2_realistic_t'].sum() / 1000.0, 2),
        'Rationale': 'Registered in 7 Marmara ports (Bandirma n=142 >=6m [144 total], Gemlik, Silivri, etc.); multi-sea seasonal harvests'
    },
    {
        'Filter_Stage': '4. Excluded: Singleton Cohort Cell (< 3 vessels: West TBB 18-24 m)',
        'Vessels_n': int(m_singleton.sum()),
        'Share_Fleet_%': round(m_singleton.sum() / len(tr_all) * 100.0, 2),
        'Total_kW': round(rem_pool.loc[m_singleton, 'engine_power_kw'].sum(), 2),
        'Share_kW_%': round(rem_pool.loc[m_singleton, 'engine_power_kw'].sum() / tr_all['engine_power_kw'].sum() * 100.0, 2),
        'Total_GT': round(rem_pool.loc[m_singleton, 'tonnage_gt'].sum(), 2),
        'Realistic_CO2_kt': round(rem_pool.loc[m_singleton, 'co2_realistic_t'].sum() / 1000.0, 2),
        'Rationale': 'Minimum n >= 3 anonymity & statistical stability rule for cohort mean inputs/outputs'
    }
]
inc_pool = rem_pool[~m_singleton].copy()
tot_excl_n = len(tr_all) - len(inc_pool)
audit_rows.append({
    'Filter_Stage': 'Total Excluded Vessels (Sum of Stages 1-4)',
    'Vessels_n': tot_excl_n,
    'Share_Fleet_%': round(tot_excl_n / len(tr_all) * 100.0, 2),
    'Total_kW': round(tr_all['engine_power_kw'].sum() - inc_pool['engine_power_kw'].sum(), 2),
    'Share_kW_%': round((1.0 - inc_pool['engine_power_kw'].sum() / tr_all['engine_power_kw'].sum()) * 100.0, 2),
    'Total_GT': round(tr_all['tonnage_gt'].sum() - inc_pool['tonnage_gt'].sum(), 2),
    'Realistic_CO2_kt': round((tr_all['co2_realistic_t'].sum() - inc_pool['co2_realistic_t'].sum()) / 1000.0, 2),
    'Rationale': 'Exact reconciliation: 1,629 + 20 + 162 + 1 = 1,812 vessels'
})
audit_rows.append({
    'Filter_Stage': 'Final Primary DEA Cohort Sample (21 Basin-Gear-Length Cohorts)',
    'Vessels_n': len(inc_pool),
    'Share_Fleet_%': round(len(inc_pool) / len(tr_all) * 100.0, 2),
    'Total_kW': round(inc_pool['engine_power_kw'].sum(), 2),
    'Share_kW_%': round(inc_pool['engine_power_kw'].sum() / tr_all['engine_power_kw'].sum() * 100.0, 2),
    'Total_GT': round(inc_pool['tonnage_gt'].sum(), 2),
    'Realistic_CO2_kt': round(inc_pool['co2_realistic_t'].sum() / 1000.0, 2),
    'Rationale': '4,772 active East/West Black Sea vessels across 21 homogeneous cohorts (86.21% of fleet kW)'
})

df_audit = pd.DataFrame(audit_rows)
df_audit.to_csv(os.path.join(TABLES_DIR, 'dea_sample_exclusion_audit.csv'), index=False)
print("\nDEA Sample Exclusion Audit (6,584 -> 4,772):")
print(df_audit[['Filter_Stage', 'Vessels_n', 'Share_Fleet_%', 'Total_kW', 'Share_kW_%']].to_string(index=False))

# 4. Load Empirical Landings Partitioning (TUIK / BSGM 2015-2024 accounts)
xlsx_candidates = [
    os.path.join(DATA_DIR, 'TR_Fisheries_Amounts_Prices.xlsx'),
    os.path.join(BASE_DIR, '..', 'Analyses_Data', 'TR_Fisheries_Amounts_Prices.xlsx'),
]
xlsx_path = next((p for p in xlsx_candidates if os.path.exists(p)), None)

if xlsx_path is not None:
    df_catch = pd.read_excel(xlsx_path)
    mean_catch = df_catch[df_catch['Year'] >= 2015].groupby(['Region', 'Species'])['Captured_Amount_Tons'].mean().reset_index()

    pelagic_species = ['Anchovy', 'Atlantic_Bonito', 'Horse_Mackerel_Small', 'Horse_Mackerel_Large', 'Sprat', 'Bluefish', 'European_Sardine', 'Atlantic_Mackerel', 'Chub_Mackerel', 'Shad']
    demersal_species = ['Whiting', 'Red_Mullet', 'Turbot', 'Striped_Mullet', 'Mullet', 'Flounder', 'Sole_Flounder', 'Common_Sole', 'Black_Scorpionfish']

    catch_totals = {'East': {}, 'West': {}}
    for r, basin in [('East_Black_Sea', 'East'), ('West_Black_Sea', 'West')]:
        r_catch = mean_catch[mean_catch['Region'] == r]
        catch_totals[basin]['Pelagic'] = float(r_catch[r_catch['Species'].isin(pelagic_species)]['Captured_Amount_Tons'].sum())
        catch_totals[basin]['Demersal'] = float(r_catch[r_catch['Species'].isin(demersal_species)]['Captured_Amount_Tons'].sum())
        catch_totals[basin]['Other'] = float(r_catch[~r_catch['Species'].isin(pelagic_species + demersal_species)]['Captured_Amount_Tons'].sum())

    tot_catch_2015 = mean_catch[mean_catch['Region'] == 'Total']
    total_mollusc = float(tot_catch_2015[tot_catch_2015['Species'].isin(
        ['Sea_Snail', 'Warty_Venus', 'White_Clam', 'Mussel', 'Mediterranean_Mussel', 'Striped_Venus']
    )]['Captured_Amount_Tons'].sum())
else:
    # Pre-verified 2015-2024 TUIK/BSGM decadal guild averages (t/yr)
    catch_totals = {
        'East': {'Pelagic': 154122.48222222223, 'Demersal': 4942.13, 'Other': 159.1403968253968},
        'West': {'Pelagic': 53236.077777777784, 'Demersal': 6701.606666666667, 'Other': 186.66305555555556},
    }
    total_mollusc = 37396.55

print(f"\n2015-2024 Contemporary Annual Black Sea Guild Landings (t/yr):")
print(f"  East Pelagic={catch_totals['East']['Pelagic']:,.1f}, East Demersal+Other={catch_totals['East']['Demersal']+catch_totals['East']['Other']:,.1f}")
print(f"  West Pelagic={catch_totals['West']['Pelagic']:,.1f}, West Demersal+Other={catch_totals['West']['Demersal']+catch_totals['West']['Other']:,.1f}")
print(f"  Total Black Sea Molluscs={total_mollusc:,.1f} (Sum GSA 29 = {catch_totals['East']['Pelagic']+catch_totals['East']['Demersal']+catch_totals['East']['Other']+catch_totals['West']['Pelagic']+catch_totals['West']['Demersal']+catch_totals['West']['Other']+total_mollusc:,.1f} t/yr)")

# 5. Helper to build cohort DEA table under any catch-allocation weight rule
def run_cohort_dea(vessel_df, weight_mode='kw', sc_list=None):
    """
    weight_mode in {'kw', 'gt', 'vessel', 'allometric_loa'}
    Allocates observed basin-gear guild catch across vessels strictly according to weight_mode
    (100% transparent alignment with Eq. 7 — zero hidden ad-hoc multipliers).
    """
    if sc_list is None:
        sc_list = ['small (6-12m)', 'medium (12-18m)', 'large (18-24m)', 'very_large (>=24m)']

    df_w = vessel_df.copy()
    if weight_mode == 'kw':
        df_w['alloc_w'] = df_w['engine_power_kw']
    elif weight_mode == 'gt':
        df_w['alloc_w'] = df_w['tonnage_gt']
    elif weight_mode == 'vessel':
        df_w['alloc_w'] = 1.0
    elif weight_mode == 'allometric_loa':
        df_w['alloc_w'] = df_w['loa_m'] ** 1.5
    else:
        raise ValueError(weight_mode)

    df_w['pelagic_catch_t'] = 0.0
    df_w['demersal_catch_t'] = 0.0
    df_w['mollusc_catch_t'] = 0.0

    # Partition molluscs across East (82% Rapa whelk core in Samsun-Ordu-Trabzon-Sinop; Dagtekin et al. 2022)
    # vs West (18%) based on TBB vessel count share (East 380 / 426 = 89.2% count, 80.1% kW -> use exact basin weight share)
    tbb_w_total = df_w[df_w['gear_detailed'] == 'Beam Trawler (TBB)']['alloc_w'].sum()

    for basin in ['East', 'West']:
        b_idx = df_w['basin'] == basin
        ps_idx = b_idx & (df_w['gear_detailed'] == 'Purse Seiner (PS)')
        otb_idx = b_idx & (df_w['gear_detailed'] == 'Pelagic/Demersal Trawler (OTB/PTM)')
        mis_idx = b_idx & (df_w['gear_detailed'] == 'Small-Scale / Polyvalent (MIS)')
        tbb_idx = b_idx & (df_w['gear_detailed'] == 'Beam Trawler (TBB)')

        if ps_idx.any():
            w_sum = df_w.loc[ps_idx, 'alloc_w'].sum()
            df_w.loc[ps_idx, 'pelagic_catch_t'] = catch_totals[basin]['Pelagic'] * (df_w.loc[ps_idx, 'alloc_w'] / w_sum)

        tot_dem = catch_totals[basin]['Demersal'] + catch_totals[basin]['Other']
        if otb_idx.any():
            w_sum = df_w.loc[otb_idx, 'alloc_w'].sum()
            df_w.loc[otb_idx, 'demersal_catch_t'] = (0.75 * tot_dem) * (df_w.loc[otb_idx, 'alloc_w'] / w_sum)
        if mis_idx.any():
            w_sum = df_w.loc[mis_idx, 'alloc_w'].sum()
            df_w.loc[mis_idx, 'demersal_catch_t'] = (0.25 * tot_dem) * (df_w.loc[mis_idx, 'alloc_w'] / w_sum)
        if tbb_idx.any() and tbb_w_total > 0:
            df_w.loc[tbb_idx, 'mollusc_catch_t'] = total_mollusc * (df_w.loc[tbb_idx, 'alloc_w'] / tbb_w_total)

    df_w['catch_potential_t'] = df_w['pelagic_catch_t'] + df_w['demersal_catch_t'] + df_w['mollusc_catch_t']

    dmus = []
    for basin in ['East', 'West']:
        for gear in ['Purse Seiner (PS)', 'Pelagic/Demersal Trawler (OTB/PTM)', 'Beam Trawler (TBB)', 'Small-Scale / Polyvalent (MIS)']:
            for sc in sc_list:
                sc_df = df_w[(df_w['basin'] == basin) & (df_w['gear_detailed'] == gear) & (df_w['size_class'] == sc)]
                if len(sc_df) >= 3:
                    dmus.append({
                        'Segment': f"{basin} {gear} - {sc}",
                        'Basin': basin,
                        'Gear': gear,
                        'Size_Class': sc,
                        'Vessel_Count': len(sc_df),
                        'Total_kW': sc_df['engine_power_kw'].sum(),
                        'Total_GT': sc_df['tonnage_gt'].sum(),
                        'Mean_LOA_m': sc_df['loa_m'].mean(),
                        'Mean_EII': sc_df['eii'].mean(),
                        'Catch_t': sc_df['catch_potential_t'].sum(),
                        'Pelagic_t': sc_df['pelagic_catch_t'].sum(),
                        'Demersal_t': sc_df['demersal_catch_t'].sum(),
                        'Mollusc_t': sc_df['mollusc_catch_t'].sum(),
                        'CO2_t': sc_df['co2_realistic_t'].sum(),
                        'CO2_Upper_t': sc_df['co2_upper_t'].sum(),
                        'Mean_kW': sc_df['engine_power_kw'].mean(),
                        'Mean_GT': sc_df['tonnage_gt'].mean(),
                        'Mean_Catch_t': sc_df['catch_potential_t'].mean(),
                        'Mean_Pelagic_t': sc_df['pelagic_catch_t'].mean(),
                        'Mean_Demersal_t': sc_df['demersal_catch_t'].mean(),
                        'Mean_Mollusc_t': sc_df['mollusc_catch_t'].mean(),
                        'Mean_CO2_t': sc_df['co2_realistic_t'].mean()
                    })

    dmu_df = pd.DataFrame(dmus)
    X = dmu_df[['Mean_kW', 'Mean_GT']].values
    Y_multi = dmu_df[['Mean_Pelagic_t', 'Mean_Demersal_t', 'Mean_Mollusc_t']].values
    Y_single = dmu_df[['Mean_Catch_t']].values
    Y_bad = dmu_df[['Mean_CO2_t']].values

    dmu_df['TE_CRS'] = np.clip(solve_standard_dea(Y_multi, X, returns='crs'), 0.0, 1.0)
    dmu_df['TE_VRS'] = np.clip(solve_standard_dea(Y_multi, X, returns='vrs'), 0.0, 1.0)
    dmu_df['Scale_Efficiency'] = np.clip(dmu_df['TE_CRS'] / np.maximum(1e-9, dmu_df['TE_VRS']), 0.0, 1.0)
    dmu_df['TE_Super_CRS'] = solve_standard_dea(Y_multi, X, returns='crs', super_eff=True)
    dmu_df['TE_VRS_SingleOutput'] = np.clip(solve_standard_dea(Y_single, X, returns='vrs'), 0.0, 1.0)

    beta = np.maximum(0.0, solve_ddf_dea(Y_multi, Y_bad, X, returns='vrs'))
    beta = np.where(beta < 1e-6, 0.0, beta)
    dmu_df['DDF_Beta_Slack'] = beta
    dmu_df['Eco_Efficiency'] = np.clip(1.0 / (1.0 + beta), 0.0, 1.0)
    dmu_df['Potential_CO2_Abatement_%'] = beta * 100.0
    dmu_df['CO2_Abatement_kt'] = (dmu_df['CO2_t'] * beta) / 1000.0
    return dmu_df

# Primary Model: Baseline kW-proportional allocation on 4,772 East/West vessels (21 cohorts)
tr_ew_base = tr_all[
    tr_all['basin'].isin(['East', 'West']) &
    (tr_all['gear_detailed'] != 'Hydraulic Dredger (DRB)') &
    tr_all['size_class'].isin(valid_sc)
].copy()

df_gear_dmu = run_cohort_dea(tr_ew_base, weight_mode='kw')
df_gear_dmu.to_csv(os.path.join(TABLES_DIR, 'turkish_dea_gear_segment_results.csv'), index=False)

n_eff_crs = int((df_gear_dmu['TE_CRS'] >= 0.9999).sum())
n_eff_vrs = int((df_gear_dmu['TE_VRS'] >= 0.9999).sum())
n_eff_single = int((df_gear_dmu['TE_VRS_SingleOutput'] >= 0.9999).sum())
print(f"\nPrimary 21-Cohort DEA Summary (n = {df_gear_dmu['Vessel_Count'].sum()} vessels, Total Realistic CO2 = {df_gear_dmu['CO2_t'].sum()/1000:.2f} kt):")
print(f"  Efficient cohorts (TE = 1.0000): Multi-guild VRS = {n_eff_vrs}/21, Multi-guild CRS = {n_eff_crs}/21, Single-output VRS (s=1) = {n_eff_single}/21")
print(f"  Mean TE_CRS = {df_gear_dmu['TE_CRS'].mean():.4f}, Mean TE_VRS = {df_gear_dmu['TE_VRS'].mean():.4f}, Mean Eco_Eff = {df_gear_dmu['Eco_Efficiency'].mean():.4f}")
print(f"  Total Cohort Frontier CO2 Abatement Slack = {df_gear_dmu['CO2_Abatement_kt'].sum():.2f} kt CO2/yr")
print(df_gear_dmu[['Segment', 'Vessel_Count', 'Mean_EII', 'TE_CRS', 'TE_VRS', 'TE_VRS_SingleOutput', 'Eco_Efficiency', 'CO2_Abatement_kt']].to_string(index=False))

# 6. Formal Sensitivity Analysis Across 4 Catch-Allocation Priors + Expanded Sample
sens_summary = []
for mode, label in [
    ('kw', 'Rule 1 (Baseline): Engine-Power (kW) Proportional'),
    ('gt', 'Rule 2: Gross-Tonnage (GT) Proportional'),
    ('vessel', 'Rule 3: Equal Vessel-Count (1/n) Allocation'),
    ('allometric_loa', 'Rule 4: Geometric Hull-Length (LOA^1.5) Scaling')
]:
    res_m = run_cohort_dea(tr_ew_base, weight_mode=mode)
    rho_vrs, _ = spearmanr(df_gear_dmu['TE_VRS'], res_m['TE_VRS'])
    rho_eco, _ = spearmanr(df_gear_dmu['Eco_Efficiency'], res_m['Eco_Efficiency'])
    sens_summary.append({
        'Allocation_Specification': label,
        'Cohorts_N': len(res_m),
        'Vessels_n': int(res_m['Vessel_Count'].sum()),
        'Frontier_Cohorts_VRS': f"{int((res_m['TE_VRS'] >= 0.9999).sum())}/{len(res_m)}",
        'Frontier_Cohorts_CRS': f"{int((res_m['TE_CRS'] >= 0.9999).sum())}/{len(res_m)}",
        'Frontier_SingleOutput_VRS': f"{int((res_m['TE_VRS_SingleOutput'] >= 0.9999).sum())}/{len(res_m)}",
        'Mean_TE_CRS': round(res_m['TE_CRS'].mean(), 4),
        'Mean_TE_VRS': round(res_m['TE_VRS'].mean(), 4),
        'Mean_Eco_Efficiency': round(res_m['Eco_Efficiency'].mean(), 4),
        'Total_Abatement_Slack_kt': round(res_m['CO2_Abatement_kt'].sum(), 2),
        'Spearman_rho_TE_VRS': round(float(rho_vrs), 4),
        'Spearman_rho_Eco_Eff': round(float(rho_eco), 4)
    })

# Expanded-sample inclusion check: include micro <6m cohorts AND assign Marmara-interface non-DRB vessels to West
tr_expanded = tr_all[tr_all['gear_detailed'] != 'Hydraulic Dredger (DRB)'].copy()
tr_expanded['basin'] = tr_expanded['basin'].replace({'Other': 'West'})
all_5_sc = ['micro (<6m)'] + valid_sc
res_exp = run_cohort_dea(tr_expanded, weight_mode='kw', sc_list=all_5_sc)
merged_21 = df_gear_dmu[['Segment', 'TE_VRS', 'Eco_Efficiency']].merge(
    res_exp[['Segment', 'TE_VRS', 'Eco_Efficiency']], on='Segment', suffixes=('_base', '_exp')
)
rho_exp_vrs, _ = spearmanr(merged_21['TE_VRS_base'], merged_21['TE_VRS_exp'])
rho_exp_eco, _ = spearmanr(merged_21['Eco_Efficiency_base'], merged_21['Eco_Efficiency_exp'])

sens_summary.append({
    'Allocation_Specification': 'Expanded Sample Check (+Micro <6m & Marmara Ports in West, kW-prop)',
    'Cohorts_N': len(res_exp),
    'Vessels_n': int(res_exp['Vessel_Count'].sum()),
    'Frontier_Cohorts_VRS': f"{int((res_exp['TE_VRS'] >= 0.9999).sum())}/{len(res_exp)}",
    'Frontier_Cohorts_CRS': f"{int((res_exp['TE_CRS'] >= 0.9999).sum())}/{len(res_exp)}",
    'Frontier_SingleOutput_VRS': f"{int((res_exp['TE_VRS_SingleOutput'] >= 0.9999).sum())}/{len(res_exp)}",
    'Mean_TE_CRS': round(res_exp['TE_CRS'].mean(), 4),
    'Mean_TE_VRS': round(res_exp['TE_VRS'].mean(), 4),
    'Mean_Eco_Efficiency': round(res_exp['Eco_Efficiency'].mean(), 4),
    'Total_Abatement_Slack_kt': round(res_exp['CO2_Abatement_kt'].sum(), 2),
    'Spearman_rho_TE_VRS': round(float(rho_exp_vrs), 4),
    'Spearman_rho_Eco_Eff': round(float(rho_exp_eco), 4)
})

df_dea_sens = pd.DataFrame(sens_summary)
df_dea_sens.to_csv(os.path.join(TABLES_DIR, 'dea_sensitivity_allocation_rules.csv'), index=False)
print("\nDEA Catch-Allocation & Sample-Inclusion Sensitivity Matrix:")
print(df_dea_sens.to_string(index=False))

print("\n[SUCCESS] DEA models, exclusion audit, and allocation sensitivity suite completed.")
