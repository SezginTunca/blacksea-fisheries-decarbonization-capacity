#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT: 06_analyze_spatial_autocorrelation.py
STUDY:  Decarbonizing Turkish Black Sea Commercial Fisheries Requires Pairing
        Industrial Engine-Power Limits with Capacity Reduction
JOURNAL: Turkish Journal of Maritime and Marine Sciences
AUTHOR: Sezgin Tunca, Ph.D. (ORCID: 0000-0001-6840-1466)
PURPOSE: Geocode all 34 registration ports of the 6,584 Turkish Black Sea commercial
         fishing vessels (27 strict Black Sea & Bosphorus coastal ports [n=6,420] +
         7 Sea of Marmara interface homeports [n=164]), aggregate fleet capacity &
         realistic duty-cycle CO2 emissions, and compute deterministic (seeded) Global
         Moran's I and Local Moran's I_i (LISA) separately for Median EII (kW/GT)
         and Total Installed Power (kW), with Benjamini-Hochberg FDR adjustments
         and Marmara-interface sensitivity checks.
INPUTS:
  - data/gfcm_blacksea_fleet.csv
OUTPUTS:
  - tables/turkish_port_level_capacity_all34.csv
  - tables/turkish_port_level_capacity.csv
  - tables/morans_i_turkiye_results.csv
  - tables/lisa_port_clusters_turkiye.csv
  - data/turkish_geocoded_ports_data.csv
  - data/moran_scatter_eii_27ports.csv
================================================================================
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TABLES_DIR = os.path.join(BASE_DIR, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

print("=== RUNNING 34-PORT GEOCODING & SPATIAL AUTOCORRELATION (MORAN'S I & LISA) ===")

# 1. Exact harbor coordinates & maritime zone classification for all 34 registration ports
PORT_CATALOG = {
    # East Black Sea Coastal Ports (13 ports)
    'HOPA':          {'lat': 41.405, 'lon': 41.433, 'zone': 'East Black Sea'},
    'PAZAR':         {'lat': 41.182, 'lon': 40.892, 'zone': 'East Black Sea'},
    'RIZE':          {'lat': 41.033, 'lon': 40.522, 'zone': 'East Black Sea'},
    'SURMENE':       {'lat': 40.912, 'lon': 40.124, 'zone': 'East Black Sea'},
    'TRABZON':       {'lat': 41.003, 'lon': 39.733, 'zone': 'East Black Sea'},
    'VAKFIKEBIR':    {'lat': 41.050, 'lon': 39.283, 'zone': 'East Black Sea'},
    'GORELE':        {'lat': 41.032, 'lon': 39.041, 'zone': 'East Black Sea'},
    'TIREBOLU':      {'lat': 41.011, 'lon': 38.820, 'zone': 'East Black Sea'},
    'GIRESUN':       {'lat': 40.918, 'lon': 38.390, 'zone': 'East Black Sea'},
    'ORDU':          {'lat': 40.984, 'lon': 37.880, 'zone': 'East Black Sea'},
    'FATSA':         {'lat': 41.031, 'lon': 37.502, 'zone': 'East Black Sea'},
    'UNYE':          {'lat': 41.130, 'lon': 37.280, 'zone': 'East Black Sea'},
    'SAMSUN':        {'lat': 41.287, 'lon': 36.330, 'zone': 'East Black Sea'},
    # West Black Sea & Bosphorus Coastal Ports (14 ports)
    'GERZE':         {'lat': 41.802, 'lon': 35.198, 'zone': 'West Black Sea'},
    'SINOP':         {'lat': 42.027, 'lon': 35.153, 'zone': 'West Black Sea'},
    'AYANCIK':       {'lat': 41.950, 'lon': 34.583, 'zone': 'West Black Sea'},
    'INEBOLU':       {'lat': 41.977, 'lon': 33.762, 'zone': 'West Black Sea'},
    'CIDE':          {'lat': 41.890, 'lon': 32.960, 'zone': 'West Black Sea'},
    'AMASRA':        {'lat': 41.748, 'lon': 32.387, 'zone': 'West Black Sea'},
    'BARTIN':        {'lat': 41.685, 'lon': 32.228, 'zone': 'West Black Sea'},
    'ZONGULDAK':     {'lat': 41.456, 'lon': 31.794, 'zone': 'West Black Sea'},
    'KDZ. EREGLI':   {'lat': 41.282, 'lon': 31.417, 'zone': 'West Black Sea'},
    'KARASU':        {'lat': 41.100, 'lon': 30.690, 'zone': 'West Black Sea'},
    'KEFKEN':        {'lat': 41.173, 'lon': 30.221, 'zone': 'West Black Sea'},
    'SILE':          {'lat': 41.176, 'lon': 29.613, 'zone': 'West Black Sea'},
    'ISTANBUL':      {'lat': 41.015, 'lon': 28.979, 'zone': 'West Black Sea (Bosphorus)'},
    'IGNEADA':       {'lat': 41.875, 'lon': 27.985, 'zone': 'West Black Sea'},
    # Sea of Marmara Interface Homeports (7 ports whose industrial fleets seasonally operate in GSA 29)
    'BANDIRMA':      {'lat': 40.353, 'lon': 27.973, 'zone': 'Marmara Interface'},
    'GEMLIK':        {'lat': 40.432, 'lon': 29.153, 'zone': 'Marmara Interface'},
    'SILIVRI':       {'lat': 41.073, 'lon': 28.246, 'zone': 'Marmara Interface'},
    'MARMARA ADASI': {'lat': 40.586, 'lon': 27.556, 'zone': 'Marmara Interface'},
    'MUDANYA':       {'lat': 40.376, 'lon': 28.883, 'zone': 'Marmara Interface'},
    'TEKIRDAG':      {'lat': 40.978, 'lon': 27.515, 'zone': 'Marmara Interface'},
    'ERDEK':         {'lat': 40.399, 'lon': 27.791, 'zone': 'Marmara Interface'}
}

fleet_df = pd.read_csv(os.path.join(DATA_DIR, 'gfcm_blacksea_fleet.csv'), low_memory=False)
tr_fleet = fleet_df[fleet_df['country'] == 'Türkiye'].copy()

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

GEAR_EF_REALISTIC = {
    'Purse Seiner (PS)': 1500.0 * 0.55 * 1.75 * 210.0 * 3.17 * 1e-6,
    'Pelagic/Demersal Trawler (OTB/PTM)': 1400.0 * 0.60 * 1.45 * 210.0 * 3.17 * 1e-6,
    'Beam Trawler (TBB)': 900.0 * 0.50 * 1.15 * 215.0 * 3.17 * 1e-6,
    'Hydraulic Dredger (DRB)': 1100.0 * 0.55 * 1.65 * 210.0 * 3.17 * 1e-6,
    'Small-Scale / Polyvalent (MIS)': 600.0 * 0.30 * 1.05 * 220.0 * 3.17 * 1e-6
}
tr_fleet['co2_realistic_t'] = tr_fleet['engine_power_kw'] * tr_fleet['gear_detailed'].map(GEAR_EF_REALISTIC)
tr_fleet['co2_upper_t'] = tr_fleet['engine_power_kw'] * 1.1616

# Verify zero unmapped ports
unmapped = set(tr_fleet['port_of_registration'].unique()) - set(PORT_CATALOG.keys())
assert len(unmapped) == 0, f"Unmapped ports found: {unmapped}"

port_stats = []
for port_name, meta in PORT_CATALOG.items():
    p_df = tr_fleet[tr_fleet['port_of_registration'] == port_name]
    n_v = len(p_df)
    if n_v == 0:
        continue
    v35 = int((p_df['vessel_age'] > 35).sum())
    port_stats.append({
        'Port': port_name,
        'Zone': meta['zone'],
        'Is_BlackSea_Coast': meta['zone'] != 'Marmara Interface',
        'Latitude': meta['lat'],
        'Longitude': meta['lon'],
        'Vessel_Count': n_v,
        'Total_kW': round(float(p_df['engine_power_kw'].sum()), 2),
        'Total_GT': round(float(p_df['tonnage_gt'].sum()), 2),
        'Mean_EII': round(float(p_df['eii'].mean()), 2),
        'Median_EII': round(float(p_df['eii'].median()), 2),
        'Mean_Age': round(float(p_df['vessel_age'].mean()), 2),
        'Vessels_Over_35': v35,
        'Pct_Over_35': round((v35 / n_v) * 100.0, 2),
        'Realistic_CO2_kt': round(float(p_df['co2_realistic_t'].sum() / 1000.0), 2),
        'UpperBound_CO2_kt': round(float(p_df['co2_upper_t'].sum() / 1000.0), 2)
    })

df_ports_all = pd.DataFrame(port_stats).sort_values(by='Total_kW', ascending=False).reset_index(drop=True)
assert df_ports_all['Vessel_Count'].sum() == 6584, "Port vessel sum must equal 6,584!"

df_ports_all.to_csv(os.path.join(TABLES_DIR, 'turkish_port_level_capacity_all34.csv'), index=False)

# Save 28-port major registry table (27 Black Sea coastal + Bandirma major hub, plus aggregated 6 minor Marmara ports row)
df_bs27 = df_ports_all[df_ports_all['Is_BlackSea_Coast']].copy().reset_index(drop=True)
df_bs28_bandirma = df_ports_all[df_ports_all['Is_BlackSea_Coast'] | (df_ports_all['Port'] == 'BANDIRMA')].copy().reset_index(drop=True)

# Keep backward-compatible filename for Figure 5 (27 strict Black Sea ports + Bandirma = 28 major ports)
df_bs28_bandirma.rename(columns={'Realistic_CO2_kt': 'CO2_Emissions_kt'}).to_csv(
    os.path.join(TABLES_DIR, 'turkish_port_level_capacity.csv'), index=False
)
df_bs28_bandirma.rename(columns={'Realistic_CO2_kt': 'CO2_Emissions_kt'}).to_csv(
    os.path.join(DATA_DIR, 'turkish_geocoded_ports_data.csv'), index=False
)

print(f"Total ports geocoded: {len(df_ports_all)} (Vessels = {df_ports_all['Vessel_Count'].sum()}, kW = {df_ports_all['Total_kW'].sum():,.2f}, Realistic CO2 = {df_ports_all['Realistic_CO2_kt'].sum():.2f} kt)")
print(f"  Strict Black Sea & Bosphorus ports: {len(df_bs27)} ports, {df_bs27['Vessel_Count'].sum()} vessels, {df_bs27['Total_kW'].sum():,.2f} kW")
print(f"  Sea of Marmara interface ports    : {len(df_ports_all)-len(df_bs27)} ports, {df_ports_all[~df_ports_all['Is_BlackSea_Coast']]['Vessel_Count'].sum()} vessels, {df_ports_all[~df_ports_all['Is_BlackSea_Coast']]['Total_kW'].sum():,.2f} kW")

# 2. Benjamini-Hochberg FDR helper
def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q_ord = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_ord, 0.0, 1.0)
    return q

# 3. Deterministic Global Moran's I across Primary 27 Black Sea Ports vs 28 Ports (+Bandirma)
def eval_global_moran(sub_df, sample_label, k_nn=4):
    gdf = gpd.GeoDataFrame(
        sub_df.copy(),
        geometry=[Point(xy) for xy in zip(sub_df['Longitude'], sub_df['Latitude'])],
        crs='EPSG:4326'
    )
    w = KNN.from_dataframe(gdf, k=k_nn)
    w.transform = 'R'

    out = []
    for col, m_label in [
        ('Median_EII', 'Engine Intensity Index (Median kW/GT)'),
        ('Mean_EII', 'Engine Intensity Index (Mean kW/GT)'),
        ('Total_kW', 'Installed Engine Power (Total kW)'),
        ('Vessel_Count', 'Registered Vessel Count (n)')
    ]:
        np.random.seed(42)
        mi = Moran(gdf[col].values, w, permutations=9999)
        out.append({
            'Sample_Scope': sample_label,
            'Ports_N': len(gdf),
            'Metric': m_label,
            'Moran_I': round(float(mi.I), 4),
            'Expected_I': round(float(mi.EI), 4),
            'Z_Score': round(float(mi.z_sim), 4),
            'p_value_perm': round(float(mi.p_sim), 4),
            'Spatial_Pattern': 'Significant Positive Clustering (p < 0.05)' if mi.p_sim < 0.05 else 'Spatially Random / Concentrated Hubs (p >= 0.05)'
        })
    return out, gdf, w

moran_27, gdf_27, w_27 = eval_global_moran(df_bs27, 'Primary: 27 Black Sea Coastal Ports (k=4)')
moran_28, gdf_28, w_28 = eval_global_moran(df_bs28_bandirma, 'Sensitivity: 28 Ports incl. Bandirma (k=4)')
moran_27_k3, _, _ = eval_global_moran(df_bs27, 'Sensitivity: 27 Black Sea Coastal Ports (k=3)', k_nn=3)

df_moran = pd.DataFrame(moran_27 + moran_28 + moran_27_k3)
df_moran.to_csv(os.path.join(TABLES_DIR, 'morans_i_turkiye_results.csv'), index=False)
print("\nGlobal Moran's I Results (Seeded, 9,999 Permutations):")
print(df_moran.to_string(index=False))

# 4. Save Standardized Spatial Lag Data for Figure S1c
np.random.seed(42)
mi_eii_27 = Moran(gdf_27['Median_EII'].values, w_27, permutations=9999)
z_eii = (gdf_27['Median_EII'].values - gdf_27['Median_EII'].mean()) / gdf_27['Median_EII'].std(ddof=1)
wz_eii = np.array([sum(w_27.weights[i][j] * z_eii[w_27.neighbors[i][j]] for j in range(len(w_27.neighbors[i]))) for i in range(len(gdf_27))])

df_scatter = pd.DataFrame({
    'Port': gdf_27['Port'].values,
    'Zone': gdf_27['Zone'].values,
    'Median_EII': gdf_27['Median_EII'].values,
    'Total_kW': gdf_27['Total_kW'].values,
    'z_Median_EII': np.round(z_eii, 4),
    'Wz_Spatial_Lag_EII': np.round(wz_eii, 4),
    'Moran_I': round(float(mi_eii_27.I), 4),
    'p_sim': round(float(mi_eii_27.p_sim), 4)
})
df_scatter.to_csv(os.path.join(DATA_DIR, 'moran_scatter_eii_27ports.csv'), index=False)

# 5. Dual LISA Analysis (Both Median_EII AND Total_kW on Primary 27 Black Sea Ports)
quad_map = {1: 'High-High', 2: 'Low-High', 3: 'Low-Low', 4: 'High-Low'}

np.random.seed(42)
lisa_eii = Moran_Local(gdf_27['Median_EII'].values, w_27, permutations=9999)
np.random.seed(42)
lisa_kw = Moran_Local(gdf_27['Total_kW'].values, w_27, permutations=9999)

q_fdr_eii = bh_fdr(lisa_eii.p_sim)
q_fdr_kw = bh_fdr(lisa_kw.p_sim)

lisa_df = gdf_27.drop(columns=['geometry']).copy()
lisa_df['EII_Local_I'] = np.round(lisa_eii.Is, 4)
lisa_df['EII_Quadrant'] = [quad_map[q] for q in lisa_eii.q]
lisa_df['EII_p_perm'] = np.round(lisa_eii.p_sim, 4)
lisa_df['EII_q_FDR'] = np.round(q_fdr_eii, 4)
lisa_df['EII_Cluster_p05'] = [
    f"{quad_map[q]} (p < 0.05)" if p < 0.05 else (f"{quad_map[q]} (Marginal p < 0.10)" if p < 0.10 else 'Not Significant')
    for q, p in zip(lisa_eii.q, lisa_eii.p_sim)
]

lisa_df['kW_Local_I'] = np.round(lisa_kw.Is, 4)
lisa_df['kW_Quadrant'] = [quad_map[q] for q in lisa_kw.q]
lisa_df['kW_p_perm'] = np.round(lisa_kw.p_sim, 4)
lisa_df['kW_q_FDR'] = np.round(q_fdr_kw, 4)
lisa_df['kW_Cluster_p05'] = [
    f"{quad_map[q]} (p < 0.05)" if p < 0.05 else (f"{quad_map[q]} (Marginal p < 0.10)" if p < 0.10 else 'Not Significant')
    for q, p in zip(lisa_kw.q, lisa_kw.p_sim)
]

lisa_df = lisa_df.sort_values(by='Total_kW', ascending=False).reset_index(drop=True)
lisa_df.to_csv(os.path.join(TABLES_DIR, 'lisa_port_clusters_turkiye.csv'), index=False)

print("\nLISA Clusters for Median EII (p < 0.10) across 27 Black Sea Coastal Ports:")
print(lisa_df[lisa_df['EII_p_perm'] < 0.10][['Port', 'Zone', 'Median_EII', 'EII_Local_I', 'EII_p_perm', 'EII_q_FDR', 'EII_Cluster_p05']].to_string(index=False))

print("\nLISA Clusters for Total kW (p < 0.10) across 27 Black Sea Coastal Ports:")
print(lisa_df[lisa_df['kW_p_perm'] < 0.10][['Port', 'Zone', 'Total_kW', 'kW_Local_I', 'kW_p_perm', 'kW_q_FDR', 'kW_Cluster_p05']].to_string(index=False))

print("\n[SUCCESS] 34-port geocoding, Global Moran's I, and dual LISA models completed.")
