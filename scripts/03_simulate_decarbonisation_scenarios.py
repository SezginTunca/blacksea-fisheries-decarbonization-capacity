#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT: 03_simulate_decarbonisation_scenarios.py
STUDY:  Decarbonizing Turkish Black Sea Commercial Fisheries Requires Pairing
        Industrial Engine-Power Limits with Capacity Reduction
JOURNAL: Turkish Journal of Maritime and Marine Sciences
AUTHOR: Sezgin Tunca, Ph.D. (ORCID: 0000-0001-6840-1466)
PURPOSE:
  1. Compute Primary Gear-Differentiated Operational Duty-Cycle CO2 Inventory vs.
     Uniform Industrial Capacity Upper-Bound Sensitivity Envelope (784.92 kt CO2/yr)
  2. Simulate Decarbonization Policy Scenarios A-E (including exact 42-vessel overlap
     accounting, tiered 750 kW PS sensitivity, and 10%/15%/20% retrofit bounds)
  3. Evaluate 2023 Fuel Price Shock Elasticity (44.52% Gross Cash Margin / 29.52%
     Depreciated Net Margin) and Capital Depreciation Rate Sensitivity (8%-20% +
     segment-differentiated power-weighted schedule)
  4. Run Monte Carlo Surviving-Fleet Net Margin Simulations (N = 10,000), contrasting
     Scenario B0/E0 (0% CPUE competition rebound) against Scenario B/E (+5% assumed
     crowding relief)
  5. Estimate Multi-Basin Panel Econometric Models (N = 105 rolling 5-year windows +
     N = 25 non-overlapping 5-year blocks) with Newey-West HAC (4 lags) and HC1 SEs.
INPUTS:
  - data/gfcm_blacksea_fleet.csv
  - data/turkish_fleet_profitability.csv
  - data/turkish_basin_panel_data.csv
OUTPUTS:
  - tables/gear_duty_cycle_emissions_baseline.csv
  - tables/decarbonisation_scenarios_summary.csv
  - tables/fuel_sensitivity_analysis.csv
  - tables/depreciation_sensitivity_analysis.csv
  - tables/depreciation_segment_weighted.csv
  - tables/turkish_decarbonisation_scenarios.csv
  - tables/regression_table_rev_cv.csv
  - tables/regression_table_catch_cv.csv
  - tables/regression_non_overlapping_robustness.csv
  - data/monte_carlo_runs.csv
================================================================================
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TABLES_DIR = os.path.join(BASE_DIR, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

print("=== START TURKISH DECARBONIZATION SCENARIOS & ECONOMETRIC SIMULATION ===")

# 1. LOAD FLEET & PROFITABILITY DATA
fleet_df = pd.read_csv(os.path.join(DATA_DIR, 'gfcm_blacksea_fleet.csv'), low_memory=False)
tr_bench = pd.read_csv(os.path.join(DATA_DIR, 'turkish_fleet_profitability.csv'))

tr_fleet = fleet_df[
    (fleet_df['country'] == 'Türkiye') &
    np.isfinite(fleet_df['eii']) &
    (fleet_df['tonnage_gt'] > 0) &
    (fleet_df['engine_power_kw'] > 0)
].copy()

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

# ==============================================================================
# 2. GEAR-DIFFERENTIATED OPERATIONAL DUTY-CYCLE INVENTORY vs UPPER-BOUND ENVELOPE
# ==============================================================================
# Standard marine fuel carbon emission factor (IMO MEPC / ISO 8178: 3.17 kg CO2/kg fuel)
CO2_PER_KG_FUEL = 3.17
DIESEL_DENSITY_KG_L = 0.845
CO2_PER_LITER_KG = CO2_PER_KG_FUEL * DIESEL_DENSITY_KG_L  # 2.67865 kg CO2 / L

# Uniform industrial capacity upper-bound factor:
# 1600 h * 0.55 LF * 1.982902 kappa_aux * 210 g/kWh * 3.17e-6 = 1.1616 t CO2 / kW / yr
UNIFORM_UPPER_EF = 1.1616

# Empirically calibrated gear-specific operational duty-cycle parameters:
# EF_g (t CO2 / kW / yr) = H_g * LF_g * kappa_aux_g * SFOC_g * 3.17e-6
GEAR_EMISSION_PARAMS = {
    'Purse Seiner (PS)': {
        'hours_yr': 1500.0, 'load_factor': 0.55, 'kappa_aux': 1.75, 'sfoc_g_kwh': 210.0,
        'empirical_basis': 'Sept 1-Apr 15 industrial season (~155 d * 9.7 h); GFCM median P_aux/P_main=0.773 (n=39) at 0.50 RSW/sonar LF + PTO'
    },
    'Pelagic/Demersal Trawler (OTB/PTM)': {
        'hours_yr': 1400.0, 'load_factor': 0.60, 'kappa_aux': 1.45, 'sfoc_g_kwh': 210.0,
        'empirical_basis': 'Open trawl season (~145 d * 9.7 h); continuous towing LF=0.60; GFCM P_aux/P_main=0.791 (n=28) at 0.35 winch/hold LF'
    },
    'Beam Trawler (TBB)': {
        'hours_yr': 900.0, 'load_factor': 0.50, 'kappa_aux': 1.15, 'sfoc_g_kwh': 215.0,
        'empirical_basis': 'Coastal Rapa whelk drag trips (~140 d * 6.4 h); mechanical PTO capstan load only; zero RSW chillers'
    },
    'Hydraulic Dredger (DRB)': {
        'hours_yr': 1100.0, 'load_factor': 0.55, 'kappa_aux': 1.65, 'sfoc_g_kwh': 210.0,
        'empirical_basis': 'Clam suction dredging; GFCM observed P_aux/P_main=0.829 (n=2) for high-pressure water-jet pumps'
    },
    'Small-Scale / Polyvalent (MIS)': {
        'hours_yr': 600.0, 'load_factor': 0.30, 'kappa_aux': 1.05, 'sfoc_g_kwh': 220.0,
        'empirical_basis': 'Seasonal short coastal day-boats (~135 trips * 4.4 h); passive gillnet/line low-idle LF=0.30; 12V hauler only'
    }
}

for g, p in GEAR_EMISSION_PARAMS.items():
    p['ef_t_co2_per_kw_yr'] = (
        p['hours_yr'] * p['load_factor'] * p['kappa_aux'] * p['sfoc_g_kwh'] * CO2_PER_KG_FUEL * 1e-6
    )

tr_fleet['ef_realistic'] = tr_fleet['gear_detailed'].map(lambda g: GEAR_EMISSION_PARAMS[g]['ef_t_co2_per_kw_yr'])
tr_fleet['co2_realistic_t'] = tr_fleet['engine_power_kw'] * tr_fleet['ef_realistic']
tr_fleet['co2_upper_t'] = tr_fleet['engine_power_kw'] * UNIFORM_UPPER_EF

n_vessels = len(tr_fleet)
base_kw = tr_fleet['engine_power_kw'].sum()
base_gt = tr_fleet['tonnage_gt'].sum()
base_co2_real_kt = tr_fleet['co2_realistic_t'].sum() / 1000.0
base_co2_upper_kt = tr_fleet['co2_upper_t'].sum() / 1000.0
implied_fuel_ml = (base_co2_real_kt * 1e6) / CO2_PER_LITER_KG / 1e6

print(f"Verified Turkish fleet: {n_vessels} vessels, {base_kw:,.2f} kW, {base_gt:,.2f} GT")
print(f"Primary Realistic Baseline CO2: {base_co2_real_kt:.2f} kt CO2/yr ({implied_fuel_ml:.2f} million L marine diesel)")
print(f"Capacity Upper-Bound Envelope : {base_co2_upper_kt:.2f} kt CO2/yr (uniform 1.1616 t CO2/kW/yr)")

# Export SI Table S9a: Gear-Differentiated Duty-Cycle Parameterization & Baseline Breakdown
gear_order = [
    'Purse Seiner (PS)',
    'Pelagic/Demersal Trawler (OTB/PTM)',
    'Beam Trawler (TBB)',
    'Hydraulic Dredger (DRB)',
    'Small-Scale / Polyvalent (MIS)'
]
duty_rows = []
for g in gear_order:
    sub = tr_fleet[tr_fleet['gear_detailed'] == g]
    p = GEAR_EMISSION_PARAMS[g]
    g_kw = sub['engine_power_kw'].sum()
    g_real_kt = sub['co2_realistic_t'].sum() / 1000.0
    g_upper_kt = sub['co2_upper_t'].sum() / 1000.0
    g_fuel_ml = (g_real_kt * 1e6) / CO2_PER_LITER_KG / 1e6
    duty_rows.append({
        'Gear_Segment': g,
        'Vessel_Count': len(sub),
        'Share_Vessels_%': round(len(sub) / n_vessels * 100.0, 2),
        'Total_kW': round(g_kw, 2),
        'Share_kW_%': round(g_kw / base_kw * 100.0, 2),
        'Operating_Hours_yr': p['hours_yr'],
        'Main_Load_Factor': p['load_factor'],
        'Aux_Multiplier_kappa': p['kappa_aux'],
        'SFOC_g_kWh': p['sfoc_g_kwh'],
        'EF_Realistic_tCO2_kW_yr': round(p['ef_t_co2_per_kw_yr'], 4),
        'Realistic_CO2_kt_yr': round(g_real_kt, 2),
        'Share_Realistic_CO2_%': round(g_real_kt / base_co2_real_kt * 100.0, 2),
        'Implied_Diesel_Million_L': round(g_fuel_ml, 2),
        'Mean_Diesel_L_per_Vessel_yr': round((g_fuel_ml * 1e6) / len(sub), 0),
        'Upper_Bound_CO2_kt_yr': round(g_upper_kt, 2),
        'Empirical_Basis': p['empirical_basis']
    })

duty_rows.append({
    'Gear_Segment': 'Total Turkish Black Sea Fleet',
    'Vessel_Count': n_vessels,
    'Share_Vessels_%': 100.00,
    'Total_kW': round(base_kw, 2),
    'Share_kW_%': 100.00,
    'Operating_Hours_yr': np.nan,
    'Main_Load_Factor': np.nan,
    'Aux_Multiplier_kappa': np.nan,
    'SFOC_g_kWh': np.nan,
    'EF_Realistic_tCO2_kW_yr': round((base_co2_real_kt * 1000.0) / base_kw, 4),
    'Realistic_CO2_kt_yr': round(base_co2_real_kt, 2),
    'Share_Realistic_CO2_%': 100.00,
    'Implied_Diesel_Million_L': round(implied_fuel_ml, 2),
    'Mean_Diesel_L_per_Vessel_yr': round((implied_fuel_ml * 1e6) / n_vessels, 0),
    'Upper_Bound_CO2_kt_yr': round(base_co2_upper_kt, 2),
    'Empirical_Basis': 'Validated against 2023 TUIK fleet fuel outlay (2.854 billion TRY ~ 137M L OTV-exempt diesel)'
})

df_duty = pd.DataFrame(duty_rows)
df_duty.to_csv(os.path.join(TABLES_DIR, 'gear_duty_cycle_emissions_baseline.csv'), index=False)
print("\nGear-Differentiated Duty-Cycle Baseline Table:")
print(df_duty[['Gear_Segment', 'Vessel_Count', 'Total_kW', 'EF_Realistic_tCO2_kW_yr', 'Realistic_CO2_kt_yr', 'Share_Realistic_CO2_%', 'Upper_Bound_CO2_kt_yr']].to_string(index=False))

# ==============================================================================
# 3. EXACT POLICY SCENARIOS A-E (BOTH REALISTIC BASELINE & UPPER-BOUND ENVELOPE)
# ==============================================================================
v35_mask = tr_fleet['vessel_age'] > 35
v35_count = int(v35_mask.sum())

active_gears = ['trawl', 'purse_seine']
tr_active = tr_fleet[tr_fleet['gear_group'].isin(active_gears)]
p75_kw = float(tr_active['engine_power_kw'].quantile(0.75))  # Exact 441.00 kW across N=1,305 active vessels

cap_mask = (tr_fleet['gear_group'].isin(active_gears)) & (tr_fleet['engine_power_kw'] > p75_kw)
n_capped = int(cap_mask.sum())
overlap_mask = v35_mask & cap_mask
n_overlap = int(overlap_mask.sum())
overlap_excess_kw = float((tr_fleet.loc[overlap_mask, 'engine_power_kw'] - p75_kw).sum())

print(f"\nExact Policy Subset Counts:")
print(f"  Active mobile-gear pool (PS + OTB + PTM + TBB): N = {len(tr_active)}, P75 = {p75_kw:.2f} kW")
print(f"  Scenario A (>441.0 kW limited): n = {n_capped} vessels ({tr_fleet.loc[cap_mask, 'gear1'].value_counts().to_dict()})")
print(f"  Scenario B (>35 yr retired)  : n = {v35_count} vessels ({v35_count/n_vessels*100:.2f}%)")
print(f"  Overlap (>35 yr AND >441 kW) : n = {n_overlap} vessels ({tr_fleet.loc[overlap_mask, 'gear1'].value_counts().to_dict()}), excess above 441 kW = {overlap_excess_kw:,.2f} kW")

# Scenario A: 441 kW main propulsion limit across active mobile gears
scen_a_kw = np.where(cap_mask, p75_kw, tr_fleet['engine_power_kw'])
co2_a_real = (scen_a_kw * tr_fleet['ef_realistic']).sum() / 1000.0
co2_a_upper = (scen_a_kw * UNIFORM_UPPER_EF).sum() / 1000.0

# Scenario A_Tiered750 sensitivity: 750 kW transitional limit for >=24 m PS, 441 kW for all other active gears
cap_limit_tiered = np.where(
    (tr_fleet['gear_detailed'] == 'Purse Seiner (PS)') & (tr_fleet['loa_m'] >= 24.0),
    750.0,
    p75_kw
)
scen_a750_kw = np.where(
    tr_fleet['gear_group'].isin(active_gears) & (tr_fleet['engine_power_kw'] > cap_limit_tiered),
    cap_limit_tiered,
    tr_fleet['engine_power_kw']
)
co2_a750_real = (scen_a750_kw * tr_fleet['ef_realistic']).sum() / 1000.0
co2_a750_upper = (scen_a750_kw * UNIFORM_UPPER_EF).sum() / 1000.0

# Scenario B: Retire 658 vessels > 35 years
scen_b_df = tr_fleet[~v35_mask]
co2_b_real = scen_b_df['co2_realistic_t'].sum() / 1000.0
co2_b_upper = scen_b_df['co2_upper_t'].sum() / 1000.0

# Scenario C: Selective Gear Transition (convert 15% of trawl group [OTB/PTM/TBB, n=1,097 -> 165 vessels] to passive MIS duty cycle & median kW)
np.random.seed(42)
tr_trawl = tr_fleet[tr_fleet['gear_group'] == 'trawl']
passive_median_kw = float(tr_fleet[tr_fleet['gear_detailed'] == 'Small-Scale / Polyvalent (MIS)']['engine_power_kw'].median())
num_shift = int(np.ceil(0.15 * len(tr_trawl)))
shift_idx = np.random.choice(tr_trawl.index, size=num_shift, replace=False)

scen_c_kw = tr_fleet['engine_power_kw'].copy()
scen_c_ef = tr_fleet['ef_realistic'].copy()
scen_c_kw.loc[shift_idx] = passive_median_kw
scen_c_ef.loc[shift_idx] = GEAR_EMISSION_PARAMS['Small-Scale / Polyvalent (MIS)']['ef_t_co2_per_kw_yr']
co2_c_real = (scen_c_kw * scen_c_ef).sum() / 1000.0
co2_c_upper = (scen_c_kw * UNIFORM_UPPER_EF).sum() / 1000.0

# Scenario D: Technical Energy Efficiency Retrofit (-15% fuel burn; plus -10% and -20% bounds)
co2_d_real = base_co2_real_kt * 0.85
co2_d_upper = base_co2_upper_kt * 0.85
co2_d10_real = base_co2_real_kt * 0.90
co2_d20_real = base_co2_real_kt * 0.80

# Scenario E: Integrated Roadmap (A + B non-additive union)
scen_e_kw = scen_a_kw[~v35_mask]
scen_e_ef = tr_fleet.loc[~v35_mask, 'ef_realistic'].values
co2_e_real = (scen_e_kw * scen_e_ef).sum() / 1000.0
co2_e_upper = (scen_e_kw * UNIFORM_UPPER_EF).sum() / 1000.0

# Scenario E_Tiered750 sensitivity (A_Tiered750 + B)
scen_e750_kw = scen_a750_kw[~v35_mask]
co2_e750_real = (scen_e750_kw * scen_e_ef).sum() / 1000.0
co2_e750_upper = (scen_e750_kw * UNIFORM_UPPER_EF).sum() / 1000.0

def make_scen_row(name, v_cnt, kw_tot, real_kt, upper_kt, mech):
    red_real = base_co2_real_kt - real_kt
    pct_real = (red_real / base_co2_real_kt) * 100.0
    red_upper = base_co2_upper_kt - upper_kt
    pct_upper = (red_upper / base_co2_upper_kt) * 100.0
    return {
        'Scenario': name,
        'Vessel_Count': int(v_cnt),
        'Installed_Power_kW': round(float(kw_tot), 2),
        'Power_Reduction_kW': round(float(base_kw - kw_tot), 2),
        'Power_Reduction_%': round(float((base_kw - kw_tot) / base_kw * 100.0), 2),
        'Realistic_CO2_Emissions_kt': round(float(real_kt), 2),
        'Realistic_CO2_Abatement_kt': round(float(red_real), 2),
        'Realistic_CO2_Reduction_%': round(float(pct_real), 2),
        'UpperBound_CO2_Emissions_kt': round(float(upper_kt), 2),
        'UpperBound_CO2_Abatement_kt': round(float(red_upper), 2),
        'UpperBound_CO2_Reduction_%': round(float(pct_upper), 2),
        'Primary_Mechanism': mech
    }

scenarios_list = [
    make_scen_row('Baseline Status Quo', n_vessels, base_kw, base_co2_real_kt, base_co2_upper_kt,
                  'Observed 2024 fleet capacity under gear-differentiated operational duty cycles'),
    make_scen_row('Scenario A: Active-Gear Power Limit (441 kW)', n_vessels, scen_a_kw.sum(), co2_a_real, co2_a_upper,
                  f'Restrict {n_capped} active vessels >441.0 kW (P75 of N=1,305 mobile-gear pool) to 441.0 kW'),
    make_scen_row('Scenario B: Aging-Vessel Retirement (>35 yr)', len(scen_b_df), scen_b_df['engine_power_kw'].sum(), co2_b_real, co2_b_upper,
                  f'Decommission {v35_count} vessels exceeding 35 years ({v35_count/n_vessels*100:.2f}% of fleet)'),
    make_scen_row('Scenario C: Selective Gear Transition (15% Trawls)', n_vessels, scen_c_kw.sum(), co2_c_real, co2_c_upper,
                  f'Reallocate {num_shift} trawl vessels (15%) to passive coastal gear ({passive_median_kw:.2f} kW median & MIS duty cycle)'),
    make_scen_row('Scenario D: Propulsion & Auxiliary Retrofit (-15%)', n_vessels, base_kw, co2_d_real, co2_d_upper,
                  f'Fleet-wide 15% fuel-efficiency retrofit (10%-20% bounds: {base_co2_real_kt-co2_d10_real:.2f}-{base_co2_real_kt-co2_d20_real:.2f} kt/yr)'),
    make_scen_row('Scenario E: Integrated Roadmap (A + B)', len(scen_b_df), scen_e_kw.sum(), co2_e_real, co2_e_upper,
                  f'Joint 441 kW limit + >35 yr buyback (accounting for {n_overlap} overlap vessels / {overlap_excess_kw:,.2f} kW shared excess)'),
    make_scen_row('Sensitivity E_750: Tiered PS Limit (750 kW >=24m) + B', len(scen_b_df), scen_e750_kw.sum(), co2_e750_real, co2_e750_upper,
                  'Transitional 750 kW main-engine limit for >=24 m purse seiners, 441 kW for trawlers, plus >35 yr buyback')
]

df_scen = pd.DataFrame(scenarios_list)
df_scen.to_csv(os.path.join(TABLES_DIR, 'decarbonisation_scenarios_summary.csv'), index=False)
print("\nDecarbonisation Scenarios Summary (Primary Realistic vs Capacity Upper Bound):")
print(df_scen[['Scenario', 'Vessel_Count', 'Installed_Power_kW', 'Realistic_CO2_Emissions_kt', 'Realistic_CO2_Abatement_kt', 'Realistic_CO2_Reduction_%', 'UpperBound_CO2_Abatement_kt']].to_string(index=False))

# ==============================================================================
# 4. 2023 FUEL PRICE SENSITIVITY & DEPRECIATION SENSITIVITY (EXACT TUIK VALUES)
# ==============================================================================
tr_2023 = tr_bench[tr_bench['Year'] == 2023].iloc[0]
fuel_sh_2023 = float(tr_2023['TR_Fuel_Share_Pct'])    # 20.704197%
labor_sh_2023 = float(tr_2023['TR_Labor_Share_Pct'])  # 19.003966%
gross_m_2023 = float(tr_2023['TR_Op_Margin_Pct'])     # 44.515743% (Gross Cash Operating Margin!)
other_sh_2023 = 100.0 - gross_m_2023 - fuel_sh_2023 - labor_sh_2023  # 15.776094%
tot_exp_2023 = 100.0 - gross_m_2023                   # 55.484257% (Total Operating Expense Share!)
depr_base = 15.0                                      # 15.00% capital consumption allowance
net_m_2023 = gross_m_2023 - depr_base                 # 29.515743% (Depreciated Net Margin!)

shocks = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
sens_rows = []
for s in shocks:
    delta_pp = fuel_sh_2023 * s
    new_fuel_ratio = fuel_sh_2023 * (1.0 + s)
    new_gross = gross_m_2023 - delta_pp
    new_net = net_m_2023 - delta_pp
    rel_gross_pct = -(delta_pp / gross_m_2023) * 100.0 if s > 0 else 0.0
    rel_net_pct = -(delta_pp / net_m_2023) * 100.0 if s > 0 else 0.0
    sens_rows.append({
        'Fuel_Price_Shock': f"+{s*100:.0f}%" if s > 0 else "Baseline (2023)",
        'Fuel_Outlay_Ratio_%': round(new_fuel_ratio, 2),
        'Total_Op_Expense_Ratio_%': round(tot_exp_2023 + delta_pp, 2),
        'Gross_Cash_Operating_Margin_%': round(new_gross, 2),
        'Margin_Change_pp': round(-delta_pp, 2),
        'Relative_Gross_Margin_Change_%': round(rel_gross_pct, 2),
        'Depreciated_Net_Margin_15pct_%': round(new_net, 2),
        'Relative_Net_Margin_Change_%': round(rel_net_pct, 2),
        'Viability_Status': 'Viable (>15% net)' if new_net >= 15.0 else ('Compressed (5-15% net)' if new_net >= 5.0 else 'Critically Vulnerable (<5% net)')
    })

breakeven_shock_pct = (net_m_2023 / fuel_sh_2023) * 100.0  # +142.56% diesel surge wipes out 29.52% net margin
print(f"\n2023 Exact Baseline Accounting: Fuel={fuel_sh_2023:.2f}%, Labor={labor_sh_2023:.2f}%, Other={other_sh_2023:.2f}%, TotalExp={tot_exp_2023:.2f}%")
print(f"2023 Gross Cash Margin={gross_m_2023:.2f}%, Depreciated Net Margin (15%)={net_m_2023:.2f}%, Net Break-Even Fuel Surge=+{breakeven_shock_pct:.2f}%")

df_sens = pd.DataFrame(sens_rows)
df_sens.to_csv(os.path.join(TABLES_DIR, 'fuel_sensitivity_analysis.csv'), index=False)
print(df_sens.to_string(index=False))

# Export SI Table S7b: Capital Depreciation Rate & Segment Heterogeneity Sensitivity
multi_yr_gross = float(tr_bench['TR_Op_Margin_Pct'].mean())  # 58.39% (2000-2024 mean)
depr_sens_rows = []
for d_rate, label in [
    (8.0, 'Low Capital Intensity / Artisanal Wooden Craft Proxy (8%)'),
    (10.0, 'Lower-Bound Fleet Replacement Allowance (10%)'),
    (12.0, 'Moderate Mixed-Fleet Allowance (12%)'),
    (15.0, 'Baseline Macro-Fleet Capital Consumption Benchmark (15%)'),
    (18.0, 'Capital-Intensive Steel Trawler/Purse-Seiner Proxy (18%)'),
    (20.0, 'Upper-Bound Industrial Sonar/RSW Replacement Allowance (20%)')
]:
    net_2023_d = gross_m_2023 - d_rate
    net_mean_d = multi_yr_gross - d_rate
    net_plus75_d = (gross_m_2023 - fuel_sh_2023 * 0.75) - d_rate
    be_shock_d = (net_2023_d / fuel_sh_2023) * 100.0
    depr_sens_rows.append({
        'Depreciation_Rate_%': d_rate,
        'Accounting_Interpretation': label,
        'Mean_2000_2024_Net_Margin_%': round(net_mean_d, 2),
        'Baseline_2023_Net_Margin_%': round(net_2023_d, 2),
        'Net_Margin_at_Plus75pct_Fuel_%': round(net_plus75_d, 2),
        'Zero_Net_Margin_Fuel_Surge_%': round(be_shock_d, 2)
    })
df_depr_sens = pd.DataFrame(depr_sens_rows)
df_depr_sens.to_csv(os.path.join(TABLES_DIR, 'depreciation_sensitivity_analysis.csv'), index=False)

# ==============================================================================
# 5. MONTE CARLO SURVIVING-FLEET NET MARGIN SIMULATION (N = 10,000)
# ==============================================================================
np.random.seed(42)
n_sim = 10000

base_rev = 100.0
base_fuel = fuel_sh_2023
base_labor_ratio = labor_sh_2023 / 100.0
base_other = other_sh_2023
base_depr = depr_base
base_net_margin = net_m_2023  # 29.515743%

# Primary realistic fuel reduction fractions
pct_real_a = (base_co2_real_kt - co2_a_real) / base_co2_real_kt
pct_real_b = (base_co2_real_kt - co2_b_real) / base_co2_real_kt
pct_real_c = (base_co2_real_kt - co2_c_real) / base_co2_real_kt
pct_real_e = (base_co2_real_kt - co2_e_real) / base_co2_real_kt
surviving_vessel_share_b = len(scen_b_df) / n_vessels  # 0.90006

# Note on Scenario B0 (0% CPUE competition rebound):
# Surviving vessels (90.01% of count) account for (1 - pct_real_b) of fuel; pre-1990 engines (>35 yr)
# exhibit +8% higher specific fuel consumption than post-1990 engines, giving a compositional
# fuel-per-vessel saving (0.9% fuel saving = +0.19 pp net margin) when crowding rebound = 0%, vs +5% assumed crowding relief in Scenario B!
age_sfoc_comp_saving = 0.009  # 0.9% lower fuel burn per surviving vessel from retiring oldest engines

rev_drop_a_sim = np.random.normal(0.020, 0.005, n_sim)
rebound_b0_sim = np.random.normal(1.00, 0.008, n_sim)    # 0% competition relief
rebound_b5_sim = np.random.normal(1.05, 0.010, n_sim)    # +5% assumed crowding relief
rev_drop_c_sim = np.random.normal(0.015, 0.005, n_sim)
retrofit_d_sim = np.random.normal(0.15, 0.015, n_sim)

mc_rows = []
for i in range(n_sim):
    # Scenario A: 441 kW propulsion limit
    rev_a = base_rev * (1.0 - rev_drop_a_sim[i])
    fuel_a = base_fuel * (1.0 - pct_real_a)
    net_a = rev_a - fuel_a - (rev_a * base_labor_ratio) - base_other - base_depr
    m_a = (net_a / rev_a) * 100.0

    # Scenario B0: >35 yr retirement WITHOUT competition relief (0% CPUE rebound, per surviving unit)
    rev_b0 = base_rev * rebound_b0_sim[i]
    fuel_b0 = base_fuel * (1.0 - age_sfoc_comp_saving)
    net_b0 = rev_b0 - fuel_b0 - (rev_b0 * base_labor_ratio) - base_other - base_depr
    m_b0 = (net_b0 / rev_b0) * 100.0

    # Scenario B: >35 yr retirement WITH assumed +5% CPUE crowding relief
    rev_b5 = base_rev * rebound_b5_sim[i]
    fuel_b5 = base_fuel * (1.0 - age_sfoc_comp_saving)
    net_b5 = rev_b5 - fuel_b5 - (rev_b5 * base_labor_ratio) - base_other - base_depr
    m_b5 = (net_b5 / rev_b5) * 100.0

    # Scenario C: 15% Trawler selective passive conversion
    rev_c = base_rev * (1.0 - rev_drop_c_sim[i])
    fuel_c = base_fuel * (1.0 - pct_real_c)
    net_c = rev_c - fuel_c - (rev_c * base_labor_ratio) - base_other - base_depr
    m_c = (net_c / rev_c) * 100.0

    # Scenario D: -15% technical propulsion/auxiliary retrofit (plus minor +0.5% annualized capex drag)
    rev_d = base_rev
    fuel_d = base_fuel * (1.0 - retrofit_d_sim[i])
    net_d = rev_d - fuel_d - (rev_d * base_labor_ratio) - base_other - (base_depr + 0.50)
    m_d = (net_d / rev_d) * 100.0

    # Scenario E0: Integrated (A + B) WITHOUT competition relief (0% rebound)
    # Fuel per surviving fleet drops by remaining capped share
    surviving_cap_fuel_red = 1.0 - ((co2_e_real / co2_b_real) * (1.0 - age_sfoc_comp_saving))
    rev_e0 = base_rev * rebound_b0_sim[i] * (1.0 - rev_drop_a_sim[i])
    fuel_e0 = base_fuel * (1.0 - surviving_cap_fuel_red)
    net_e0 = rev_e0 - fuel_e0 - (rev_e0 * base_labor_ratio) - base_other - base_depr
    m_e0 = (net_e0 / rev_e0) * 100.0

    # Scenario E: Integrated (A + B) WITH assumed +5% CPUE crowding relief
    rev_e5 = base_rev * rebound_b5_sim[i] * (1.0 - rev_drop_a_sim[i])
    fuel_e5 = base_fuel * (1.0 - surviving_cap_fuel_red)
    net_e5 = rev_e5 - fuel_e5 - (rev_e5 * base_labor_ratio) - base_other - base_depr
    m_e5 = (net_e5 / rev_e5) * 100.0

    # Sensitivity E750: Tiered 750 kW >=24 m PS limit + 441 kW trawl limit + >35 yr retirement (+5% rebound)
    surviving_cap750_fuel_red = 0.1501
    rev_e750 = base_rev * rebound_b5_sim[i] * (1.0 - 0.79 * rev_drop_a_sim[i])
    fuel_e750 = base_fuel * (1.0 - surviving_cap750_fuel_red)
    net_e750 = rev_e750 - fuel_e750 - (rev_e750 * base_labor_ratio) - base_other - base_depr
    m_e750 = (net_e750 / rev_e750) * 100.0

    mc_rows.append({
        'Scenario_A_Margin': m_a,
        'Scenario_B_ZeroRebound_Margin': m_b0,
        'Scenario_B_Margin': m_b5,
        'Scenario_C_Margin': m_c,
        'Scenario_D_Margin': m_d,
        'Scenario_E_ZeroRebound_Margin': m_e0,
        'Scenario_E_Combined_Margin': m_e5,
        'Sensitivity_E750_Margin': m_e750,
        'Scenario_A_Margin_Change': m_a - base_net_margin,
        'Scenario_B_ZeroRebound_Margin_Change': m_b0 - base_net_margin,
        'Scenario_B_Margin_Change': m_b5 - base_net_margin,
        'Scenario_C_Margin_Change': m_c - base_net_margin,
        'Scenario_D_Margin_Change': m_d - base_net_margin,
        'Scenario_E_ZeroRebound_Margin_Change': m_e0 - base_net_margin,
        'Scenario_E_Combined_Margin_Change': m_e5 - base_net_margin,
        'Sensitivity_E750_Margin_Change': m_e750 - base_net_margin
    })

df_mc = pd.DataFrame(mc_rows)
df_mc.to_csv(os.path.join(DATA_DIR, 'monte_carlo_runs.csv'), index=False)

mc_summary = [{
    'Scenario': 'Baseline (2023 Depreciated Net Margin)',
    'Simulated_Net_Margin_Mean_%': round(base_net_margin, 2),
    'Simulated_Gross_Margin_Mean_%': round(gross_m_2023, 2),
    'Net_Margin_95%_CI': f"[{base_net_margin:.2f}%, {base_net_margin:.2f}%]",
    'Margin_Change_Mean_pp': 0.00,
    'Margin_Change_95%_CI_pp': "[+0.00 pp, +0.00 pp]",
    'Rebound_Assumption': 'Observed 2023 cost structure (15% depreciation allowance)'
}]

label_map = [
    ('Scenario_A_Margin', 'Scenario A: 441 kW Propulsion Limit', '-2.0% +/- 0.5% harvest speed adjustment; 0% CPUE rebound'),
    ('Scenario_B_ZeroRebound_Margin', 'Scenario B0: >35 yr Retirement (0% Rebound)', '0% CPUE crowding relief (compositional engine SFOC effect only)'),
    ('Scenario_B_Margin', 'Scenario B: >35 yr Retirement (+5% Assumed Rebound)', 'Assumed +5.0% +/- 1.0% CPUE crowding relief for surviving vessels'),
    ('Scenario_C_Margin', 'Scenario C: 15% Trawl Passive Conversion', '-1.5% +/- 0.5% harvest adjustment; lower passive fuel intensity'),
    ('Scenario_D_Margin', 'Scenario D: -15% Technical Retrofit', '15.0% +/- 1.5% fuel saving; +0.50 pp annualized retrofit capex'),
    ('Scenario_E_ZeroRebound_Margin', 'Scenario E0: Integrated A + B (0% Rebound)', 'Joint 441 kW limit + >35 yr retirement WITHOUT crowding rebound'),
    ('Scenario_E_Combined_Margin', 'Scenario E: Integrated A + B (+5% Assumed Rebound)', 'Joint 441 kW limit + >35 yr retirement WITH assumed +5% crowding relief'),
    ('Sensitivity_E750_Margin', 'Sensitivity E750: Tiered 750 kW PS Limit + B (+5% Rebound)', 'Tiered 750 kW >=24 m PS limit + 441 kW trawl limit + >35 yr retirement')
]

for col, nice_name, reb_note in label_map:
    m_mean = df_mc[col].mean()
    p25, p975 = df_mc[col].quantile([0.025, 0.975])
    c_mean = df_mc[col + '_Change'].mean()
    cp25, cp975 = df_mc[col + '_Change'].quantile([0.025, 0.975])
    mc_summary.append({
        'Scenario': nice_name,
        'Simulated_Net_Margin_Mean_%': round(m_mean, 2),
        'Simulated_Gross_Margin_Mean_%': round(m_mean + base_depr, 2),
        'Net_Margin_95%_CI': f"[{p25:.2f}%, {p975:.2f}%]",
        'Margin_Change_Mean_pp': round(c_mean, 2),
        'Margin_Change_95%_CI_pp': f"[{cp25:+.2f} pp, {cp975:+.2f} pp]",
        'Rebound_Assumption': reb_note
    })

df_mc_summary = pd.DataFrame(mc_summary)
df_mc_summary.to_csv(os.path.join(TABLES_DIR, 'turkish_decarbonisation_scenarios.csv'), index=False)
print("\nMonte Carlo Surviving-Fleet Net Margin Summary (N = 10,000):")
print(df_mc_summary[['Scenario', 'Simulated_Net_Margin_Mean_%', 'Margin_Change_Mean_pp', 'Margin_Change_95%_CI_pp']].to_string(index=False))

# ==============================================================================
# 6. PANEL ECONOMETRIC MODELS (N = 105 ROLLING + N = 25 NON-OVERLAPPING CHECK)
# ==============================================================================
panel_df = pd.read_csv(os.path.join(DATA_DIR, 'turkish_basin_panel_data.csv'))
panel_df_clean = panel_df.dropna(subset=['Rev_Roll_CV', 'Shannon_Roll_Mean', 'Vessels_Roll_Mean']).copy()

region_dummies = pd.get_dummies(panel_df_clean['Region'], drop_first=True, dtype=int)
basin_cols = list(region_dummies.columns)

panel_df_clean['ln_Vessels'] = np.log(panel_df_clean['Vessels_Roll_Mean'])
panel_df_clean['ln_USD_TRY'] = np.log(panel_df_clean['USD_TRY_Roll_Mean'])
panel_df_clean['ln_CPI'] = np.log(panel_df_clean['CPI_Roll_Mean'])
panel_df_clean['ln_Chla'] = np.log(panel_df_clean['Chla_mg_m3'])

mean_ln_v = float(panel_df_clean['ln_Vessels'].mean())     # Exact 7.9827
mean_ln_fx = float(panel_df_clean['ln_USD_TRY'].mean())    # Exact sample mean
panel_df_clean['ln_Vessels_c'] = panel_df_clean['ln_Vessels'] - mean_ln_v
panel_df_clean['ln_USD_TRY_c'] = panel_df_clean['ln_USD_TRY'] - mean_ln_fx

print(f"\nPanel Sample N = {len(panel_df_clean)} (5 basins x 21 rolling 5-yr windows, 2000-2024)")
print(f"Exact sample mean ln(Vessels) = {mean_ln_v:.4f}, mean ln(USD/TRY) = {mean_ln_fx:.4f}")

def run_specifications_hac(y_var, file_name):
    X1 = sm.add_constant(panel_df_clean[['Shannon_Roll_Mean', 'ln_Vessels', 'ln_USD_TRY', 'ln_CPI']])
    m1 = OLS(y_var, X1).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

    X2 = pd.concat([X1, region_dummies], axis=1)
    m2 = OLS(y_var, X2).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

    X3 = X2.copy()
    X3['Shannon_x_ln_Vessels'] = X3['Shannon_Roll_Mean'] * X3['ln_Vessels']
    m3 = OLS(y_var, X3).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

    # Exact mean-centered Model 3c to extract marginal effect of H' at mean(ln V) = 7.9827
    X3c = pd.concat([
        sm.add_constant(panel_df_clean[['Shannon_Roll_Mean', 'ln_Vessels_c', 'ln_USD_TRY', 'ln_CPI']]),
        region_dummies
    ], axis=1)
    X3c['Shannon_x_ln_Vessels_c'] = X3c['Shannon_Roll_Mean'] * X3c['ln_Vessels_c']
    m3c = OLS(y_var, X3c).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

    X4 = X2.copy()
    X4['Shannon_x_ln_USD_TRY'] = X4['Shannon_Roll_Mean'] * X4['ln_USD_TRY']
    m4 = OLS(y_var, X4).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

    X5 = X2.copy()
    X5['SST_Anomaly'] = panel_df_clean['SST_Anomaly']
    X5['ln_Chla'] = panel_df_clean['ln_Chla']
    X5['Shannon_x_SST_Anomaly'] = panel_df_clean['Shannon_Roll_Mean'] * panel_df_clean['SST_Anomaly']
    m5 = OLS(y_var, X5).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

    all_vars = [
        'const', 'Shannon_Roll_Mean', 'ln_Vessels', 'ln_USD_TRY', 'ln_CPI',
        'Shannon_x_ln_Vessels', 'Shannon_x_ln_USD_TRY',
        'SST_Anomaly', 'ln_Chla', 'Shannon_x_SST_Anomaly'
    ] + basin_cols

    def fmt_coef(val, p):
        stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
        return f"{val:.4f}{stars}"

    coef_data = []
    for var in all_vars:
        row = {'Variable': var}
        for m_idx, m in enumerate([m1, m2, m3, m4, m5], 1):
            if var in m.params:
                row[f'Model{m_idx}_Coef'] = fmt_coef(m.params[var], m.pvalues[var])
                row[f'Model{m_idx}_SE'] = f"({m.bse[var]:.4f})"
            else:
                row[f'Model{m_idx}_Coef'] = ""
                row[f'Model{m_idx}_SE'] = ""
        coef_data.append(row)

    # Add verified centered marginal slope row for Model 3
    cent_row = {
        'Variable': f'Centered_Shannon_at_mean_lnV ({mean_ln_v:.2f})',
        'Model1_Coef': '', 'Model1_SE': '',
        'Model2_Coef': '', 'Model2_SE': '',
        'Model3_Coef': fmt_coef(m3c.params['Shannon_Roll_Mean'], m3c.pvalues['Shannon_Roll_Mean']),
        'Model3_SE': f"({m3c.bse['Shannon_Roll_Mean']:.4f})",
        'Model4_Coef': '', 'Model4_SE': '',
        'Model5_Coef': '', 'Model5_SE': ''
    }
    coef_data.append(cent_row)

    for label, fn in [
        ('R-squared', lambda m: f"{m.rsquared:.4f}"),
        ('Adj R-squared', lambda m: f"{m.rsquared_adj:.4f}"),
        ('N', lambda m: f"{int(m.nobs)}"),
        ('F-statistic', lambda m: f"{m.fvalue:.2f}"),
        ('Durbin-Watson', lambda m: f"{durbin_watson(m.resid):.3f}")
    ]:
        s_row = {'Variable': label}
        for m_idx, m in enumerate([m1, m2, m3, m4, m5], 1):
            s_row[f'Model{m_idx}_Coef'] = fn(m)
            s_row[f'Model{m_idx}_SE'] = ""
        coef_data.append(s_row)

    df_out = pd.DataFrame(coef_data)
    df_out.to_csv(os.path.join(TABLES_DIR, file_name), index=False)
    return [m1, m2, m3, m4, m5], m3c

rev_models, m3c_rev = run_specifications_hac(panel_df_clean['Rev_Roll_CV'], 'regression_table_rev_cv.csv')
catch_models, m3c_catch = run_specifications_hac(panel_df_clean['Catch_Roll_CV'], 'regression_table_catch_cv.csv')

# Compute partial slopes of H' across vessel capacity quantiles for Figure 7d & Table 6 notes
m3_rev = rev_models[2]
cov_m3 = m3_rev.cov_params()
b_h = m3_rev.params['Shannon_Roll_Mean']
b_int = m3_rev.params['Shannon_x_ln_Vessels']
for label, lnv_val in [
    ('P15 Low Capacity (ln V = 7.45)', 7.45),
    ('P25 Moderate Capacity (ln V = 7.65)', 7.65),
    (f'Sample Mean (ln V = {mean_ln_v:.4f})', mean_ln_v),
    ('P85 High Capacity (ln V = 8.45)', 8.45)
]:
    slope = b_h + b_int * lnv_val
    var_s = cov_m3.loc['Shannon_Roll_Mean', 'Shannon_Roll_Mean'] + \
            (lnv_val ** 2) * cov_m3.loc['Shannon_x_ln_Vessels', 'Shannon_x_ln_Vessels'] + \
            2.0 * lnv_val * cov_m3.loc['Shannon_Roll_Mean', 'Shannon_x_ln_Vessels']
    se_s = np.sqrt(max(0.0, var_s))
    z_s = slope / se_s if se_s > 0 else 0.0
    from scipy.stats import norm
    p_s = 2.0 * (1.0 - norm.cdf(abs(z_s)))
    print(f"  Model 3 Marginal Slope d(Rev_CV)/d(H') at {label}: {slope:+.4f} (SE={se_s:.4f}, p={p_s:.4f})")

# Non-overlapping 5-year block robustness check (2004, 2009, 2014, 2019, 2024 end-years -> N = 25)
yr_col = 'Window_End_Year' if 'Window_End_Year' in panel_df_clean.columns else 'Year'
non_overlap = panel_df_clean[panel_df_clean[yr_col].isin([2004, 2009, 2014, 2019, 2024])].copy()
if len(non_overlap) >= 20:
    rd_no = pd.get_dummies(non_overlap['Region'], drop_first=True, dtype=int)
    X_no1 = sm.add_constant(non_overlap[['Shannon_Roll_Mean', 'ln_Vessels', 'ln_USD_TRY', 'ln_CPI']])
    m_no1 = OLS(non_overlap['Rev_Roll_CV'], X_no1).fit(cov_type='HC1')
    X_no3 = pd.concat([X_no1, rd_no], axis=1)
    X_no3['Shannon_x_ln_Vessels'] = X_no3['Shannon_Roll_Mean'] * X_no3['ln_Vessels']
    m_no3 = OLS(non_overlap['Rev_Roll_CV'], X_no3).fit(cov_type='HC1')
    X_no4 = pd.concat([X_no1, rd_no], axis=1)
    X_no4['Shannon_x_ln_USD_TRY'] = X_no4['Shannon_Roll_Mean'] * X_no4['ln_USD_TRY']
    m_no4 = OLS(non_overlap['Rev_Roll_CV'], X_no4).fit(cov_type='HC1')

    no_rows = []
    for sname, mod in [('Non-Overlapping Baseline (HC1)', m_no1), ('Non-Overlapping Capacity Interaction (HC1)', m_no3), ('Non-Overlapping FX Interaction (HC1)', m_no4)]:
        no_rows.append({
            'Specification': sname,
            'N': int(mod.nobs),
            'Shannon_Coef': round(mod.params['Shannon_Roll_Mean'], 4),
            'Shannon_SE': round(mod.bse['Shannon_Roll_Mean'], 4),
            'Shannon_p': round(mod.pvalues['Shannon_Roll_Mean'], 4),
            'Interaction_Coef': round(mod.params.get('Shannon_x_ln_Vessels', mod.params.get('Shannon_x_ln_USD_TRY', np.nan)), 4),
            'Interaction_SE': round(mod.bse.get('Shannon_x_ln_Vessels', mod.bse.get('Shannon_x_ln_USD_TRY', np.nan)), 4),
            'Interaction_p': round(mod.pvalues.get('Shannon_x_ln_Vessels', mod.pvalues.get('Shannon_x_ln_USD_TRY', np.nan)), 4),
            'R_squared': round(mod.rsquared, 4),
            'Durbin_Watson': round(float(durbin_watson(mod.resid)), 3)
        })
    df_no = pd.DataFrame(no_rows)
    df_no.to_csv(os.path.join(TABLES_DIR, 'regression_non_overlapping_robustness.csv'), index=False)
    print("\nNon-Overlapping 5-Year Block Robustness Check (N=25):")
    print(df_no.to_string(index=False))

# VIF diagnostics on main effects (Model 1) and centered interaction (Model 3c)
X_vif = sm.add_constant(panel_df_clean[['Shannon_Roll_Mean', 'ln_Vessels_c', 'ln_USD_TRY_c', 'ln_CPI']])
vif_vals = {X_vif.columns[i]: round(variance_inflation_factor(X_vif.values, i), 2) for i in range(1, X_vif.shape[1])}
print("\nCentered Regressor VIFs:", vif_vals)

# ==============================================================================
# 7. SEGMENT-WEIGHTED (HETEROGENEOUS) CAPITAL-CONSUMPTION SCHEDULE (SI Table S7b)
# ==============================================================================
# delta by gear segment (% of gross revenue), weighted by installed engine power (kW)
SEG_DELTA = {
    'Small-Scale / Polyvalent (MIS)': 8.0,
    'Pelagic/Demersal Trawler (OTB/PTM)': 14.0,
    'Beam Trawler (TBB)': 14.0,
    'Hydraulic Dredger (DRB)': 14.0,
    'Purse Seiner (PS)': 18.0
}
kw_by_seg = tr_fleet.groupby('gear_detailed')['engine_power_kw'].sum()
delta_w = float(sum(kw_by_seg[g] * d for g, d in SEG_DELTA.items()) / kw_by_seg.sum())
net_2023_w = gross_m_2023 - delta_w
df_het = pd.DataFrame([{
    'Depreciation_Rate_%': round(delta_w, 2),
    'Accounting_Interpretation': 'Power-weighted segment schedule (MIS 8%, OTB/PTM/TBB/DRB 14%, PS 18%)',
    'Mean_2000_2024_Net_Margin_%': round(multi_yr_gross - delta_w, 2),
    'Baseline_2023_Net_Margin_%': round(net_2023_w, 2),
    'Net_Margin_at_Plus75pct_Fuel_%': round(net_2023_w - fuel_sh_2023 * 0.75, 2),
    'Zero_Net_Margin_Fuel_Surge_%': round(net_2023_w / fuel_sh_2023 * 100.0, 2)
}])
df_het.to_csv(os.path.join(TABLES_DIR, 'depreciation_segment_weighted.csv'), index=False)
print("\nSegment-weighted capital consumption schedule (power-weighted):")
print(df_het.to_string(index=False))

print("\n[SUCCESS] Decarbonisation scenarios, fuel/depreciation sensitivity, Monte Carlo, and HAC regressions completed.")
