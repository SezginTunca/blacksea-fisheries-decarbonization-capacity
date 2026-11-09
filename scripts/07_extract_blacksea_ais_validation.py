#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT: 07_extract_blacksea_ais_validation.py
STUDY:  Decarbonizing Turkish Black Sea Commercial Fisheries Requires Pairing
        Industrial Engine-Power Limits with Capacity Reduction
JOURNAL: Turkish Journal of Maritime and Marine Sciences
AUTHOR: Sezgin Tunca, Ph.D. (ORCID: 0000-0001-6840-1466)
PURPOSE: Extract illustrative satellite AIS operational speed check metrics for
         commercial fishing vessels operating in the Black Sea (2018-2024) from
         Global Fishing Watch (GFW) data, cross-reference ITU call signs (IRCS)
         against GFCM GSA 29 registry records to restore original Turkish vessel
         names anglicized by GFW's transponder ingestion pipeline, and render
         Supplementary Figure S2 at 600 DPI.
INPUTS:
  - data/gfw_blacksea_ais_events_20vessels.csv (or GFW_EVENTS_CSV / GFW_REGISTRY_CSV)
  - data/gfcm_blacksea_fleet.csv
OUTPUTS:
  - tables/blacksea_ais_vessel_telemetry_validation.csv (Supplementary Table S11)
  - figures/figS2_ais_speed_validation.png & .pdf
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
GFW_DATA_CSV = os.environ.get(
    "GFW_EVENTS_CSV",
    os.path.join(DATA_DIR, "GFW_Turkish_Waters_Fishing_Events_Master.csv"),
)
GFW_REG_CSV = os.environ.get(
    "GFW_REGISTRY_CSV",
    os.path.join(DATA_DIR, "GFW_Turkish_Fleet_Registry_Master.csv"),
)
GFCM_FLEET_CSV = os.path.join(DATA_DIR, "gfcm_blacksea_fleet.csv")
GFW_CACHE_CSV = os.path.join(DATA_DIR, "gfw_blacksea_ais_events_20vessels.csv")
TABLES_DIR = os.path.join(BASE_DIR, "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("STEP 07: Ingesting GFW Fishing Events, GFW Registry & GFCM GSA 29 Fleet")
print("=" * 80)

if os.path.exists(GFW_DATA_CSV):
    df_gfw = pd.read_csv(GFW_DATA_CSV)
    df_reg = pd.read_csv(GFW_REG_CSV) if os.path.exists(GFW_REG_CSV) else pd.DataFrame()
    df_bs = df_gfw[df_gfw["basin"] == "Black Sea"].copy()
    if len(df_reg) > 0:
        reg_cols = df_reg[["mmsi", "callsign", "flag"]].drop_duplicates("mmsi")
        df_bs = df_bs.merge(reg_cols, on="mmsi", how="left")
    df_bs.to_csv(GFW_CACHE_CSV, index=False)
else:
    print("Using pre-extracted Black Sea AIS subset in data/gfw_blacksea_ais_events_20vessels.csv")
    df_gfw = pd.read_csv(GFW_CACHE_CSV)
    df_reg = df_gfw[["mmsi", "callsign", "flag"]].drop_duplicates("mmsi") if "callsign" in df_gfw.columns else pd.DataFrame()
df_gfcm = pd.read_csv(GFCM_FLEET_CSV, low_memory=False)
df_gfcm["ircs_clean"] = df_gfcm["ircs_if_applicable"].astype(str).str.strip().str.upper()

bs_events = df_gfw[df_gfw["basin"] == "Black Sea"].copy()
print(f"Total GFW Fishing Events Loaded: {len(df_gfw):,}")
print(f"Black Sea Fishing Events: {len(bs_events):,} across {bs_events['vessel_name'].nunique()} AIS-tracked vessels")

# Original Turkish registry names restored from ITU Call Sign (IRCS) / Turkish Maritime Registry
# Note: GFW's automated AIS shipname parser translates Turkish maritime terms into English
# (e.g., BALIKÇILIK -> FISHERIES, KARDEŞLER -> BROTHERS, REİS/KAPTAN -> MASTER/CAPTAIN, AVCI BABA -> HUNTER FATHER).
TURKISH_NAME_RESTORATION = {
    "YUSUF DEMIROGLU": "YUSUF DEMİROĞLU",
    "KOROGLU FISHERIES A": "KÖROĞLU BALIKÇILIK A",
    "BURAKYAGIZ": "BURAK YAĞIZ (GEO Charter)",
    "AHMET MOLLAOGLU 1": "AHMET MOLLAOĞLU 1",
    "SEA BASARAN-1": "DENİZ BAŞARAN-1 (MALKOÇOĞLU ALİ KAPTAN)",
    "HUNTER FATHER": "AVCI BABA",
    "REFIK MASTER": "REFİK REİS",
    "ZOTI AHMET CAPTAIN 5": "ZOTİ AHMET KAPTAN 5",
    "YAKUPOGLU-II": "YAKUPOĞLU-II",
    "CINAR BROTHERS": "ÇINAR KARDEŞLER",
    "CINAR IBRAHIM": "ÇINAR İBRAHİM",
    "BURHAN CAPTAIN 5": "BURHAN KAPTAN 5",
    "KORKMAZLAR FISHERIES": "KORKMAZLAR BALIKÇILIK",
    "CALISKAN BROTHERS": "ÇALIŞKAN KARDEŞLER",
    "ADALI BROTHERS": "ADALI KARDEŞLER",
    "ORHAN CAPTAIN 2": "ORHAN KAPTAN 2",
    "HICAZ FISHERIES": "HİCAZ BALIKÇILIK",
    "GENC FISHERIES A": "GENÇ BALIKÇILIK A (KARAİNCİ 53)",
    "DUA": "DUA",
    "AYDIN TOKER": "AYDIN TOKER",
}


def clean_gear_label(gear_str):
    g = str(gear_str).upper()
    if "TRAWLER" in g:
        return "Demersal / Midwater Trawler"
    elif "PURSE" in g:
        return "Industrial Purse Seiner"
    else:
        return "Polyvalent / Unclassified AIS"


bs_events["gear_clean"] = bs_events["gear_type"].apply(clean_gear_label)

reg_lookup = {}
if len(df_reg) > 0:
    for _, rr in df_reg.iterrows():
        reg_lookup[int(rr["mmsi"])] = (
            str(rr.get("callsign", "")).strip() if pd.notna(rr.get("callsign")) else "—",
            str(rr.get("flag", "TUR")).strip() if pd.notna(rr.get("flag")) else "TUR",
        )

summary_rows = []
for vessel, group in bs_events.groupby("vessel_name"):
    n_events = len(group)
    gear = group["gear_clean"].mode()[0] if len(group["gear_clean"].mode()) > 0 else "Polyvalent / Unclassified AIS"
    mmsi = int(group["mmsi"].iloc[0]) if pd.notna(group["mmsi"].iloc[0]) else 0
    callsign, flag_code = reg_lookup.get(mmsi, ("—", "TUR"))

    # Match IRCS or exact name in GSA 29 primary register
    gfcm_hit = df_gfcm[df_gfcm["ircs_clean"] == callsign.upper()] if callsign != "—" else pd.DataFrame()
    if len(gfcm_hit) == 0 and vessel == "AHMET MOLLAOGLU 1":
        gfcm_hit = df_gfcm[df_gfcm["vessel_name"].astype(str).str.strip().str.upper() == "AHMET MOLLAOĞLU 1"]
    if len(gfcm_hit) > 0:
        h0 = gfcm_hit.iloc[0]
        reg_status = f"GSA 29 ({h0['port_of_registration']}, {h0['gear1']}, {h0['loa_m']:.1f} m)"
    elif flag_code == "GEO":
        reg_status = "Georgian Charter Flag (MMSI 213)"
    else:
        reg_status = "Seasonal GSA 28/22 Industrial Permit"

    years_str = (
        f"{group['year'].min()}–{group['year'].max()}"
        if group["year"].min() != group["year"].max()
        else str(group["year"].min())
    )
    lon_range = f"{group['longitude'].min():.1f}°E–{group['longitude'].max():.1f}°E"

    summary_rows.append(
        {
            "turkish_vessel_name": TURKISH_NAME_RESTORATION.get(vessel, vessel),
            "gfw_ais_name": vessel,
            "mmsi": mmsi,
            "ircs_callsign": callsign,
            "flag": flag_code,
            "registry_linkage": reg_status,
            "gear_category": gear,
            "n_events": n_events,
            "survey_years": years_str,
            "mean_speed_knots": round(group["avg_speed_knots"].mean(), 2),
            "median_speed_knots": round(group["avg_speed_knots"].median(), 2),
            "max_speed_knots": round(group["avg_speed_knots"].max(), 2),
            "mean_distance_shore_km": round(group["distance_shore_km"].mean(), 2),
            "mean_distance_port_km": round(group["distance_port_km"].mean(), 2),
            "operational_longitude_range": lon_range,
        }
    )

df_table_s11 = pd.DataFrame(summary_rows).sort_values("n_events", ascending=False).reset_index(drop=True)
out_csv_path = os.path.join(TABLES_DIR, "blacksea_ais_vessel_telemetry_validation.csv")
df_table_s11.to_csv(out_csv_path, index=False)
print(f"Saved enriched Table S11 ({len(df_table_s11)} vessels, {df_table_s11['n_events'].sum()} events) -> {out_csv_path}")

trawler_events = bs_events[bs_events["gear_clean"] == "Demersal / Midwater Trawler"]
ps_events = bs_events[bs_events["gear_clean"] == "Industrial Purse Seiner"]
poly_events = bs_events[bs_events["gear_clean"] == "Polyvalent / Unclassified AIS"]

print(
    f"Event breakdown: Total={len(bs_events)} | PS={len(ps_events)} (mean {ps_events['avg_speed_knots'].mean():.2f} kn, med {ps_events['avg_speed_knots'].median():.2f} kn) "
    f"| Trawl={len(trawler_events)} (mean {trawler_events['avg_speed_knots'].mean():.2f} kn, med {trawler_events['avg_speed_knots'].median():.2f} kn, IQR {trawler_events['avg_speed_knots'].quantile(0.25):.2f}-{trawler_events['avg_speed_knots'].quantile(0.75):.2f} kn) "
    f"| Poly={len(poly_events)} (mean {poly_events['avg_speed_knots'].mean():.2f} kn)"
)
print(f"Overall mean active event speed: {bs_events['avg_speed_knots'].mean():.2f} kn (median {bs_events['avg_speed_knots'].median():.2f} kn, max {bs_events['avg_speed_knots'].max():.2f} kn)")

# Render Figure S2 (600 DPI)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10.5,
    "axes.titlesize": 11.8,
    "axes.labelsize": 11.0,
    "xtick.labelsize": 9.8,
    "ytick.labelsize": 9.8,
    "legend.fontsize": 8.8,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)

ax1 = axes[0]
bins = np.linspace(0, 9, 37)
ax1.hist(
    ps_events["avg_speed_knots"],
    bins=bins,
    color="#0284C7",
    alpha=0.62,
    label=f"Purse Seiners (13 vessels, n = {len(ps_events)} events)",
    density=True,
    edgecolor="white",
    linewidth=0.5,
)
ax1.hist(
    trawler_events["avg_speed_knots"],
    bins=bins,
    color="#DC2626",
    alpha=0.68,
    label=f"Trawler Tracked (1 vessel, n = {len(trawler_events)} events)",
    density=True,
    edgecolor="white",
    linewidth=0.5,
)
ax1.axvspan(2.5, 3.5, color="#15803D", alpha=0.18, label="Assumed Towing Window (2.5–3.5 kn)")
ax1.axvline(
    trawler_events["avg_speed_knots"].mean(),
    color="#B91C1C",
    linestyle="--",
    linewidth=1.5,
    label=f"Observed Trawler Mean ({trawler_events['avg_speed_knots'].mean():.2f} kn)",
)
ax1.axvline(
    ps_events["avg_speed_knots"].mean(),
    color="#0369A1",
    linestyle=":",
    linewidth=1.5,
    label=f"Observed Purse Seine Mean ({ps_events['avg_speed_knots'].mean():.2f} kn)",
)

ax1.set_xlabel("Observed AIS Fishing-Event Speed Over Ground (knots)", fontweight="bold")
ax1.set_ylabel("Probability Density", fontweight="bold")
ax1.set_title("(a) Illustrative AIS Fishing-Event Speeds (20 Vessels, 849 Events)", pad=8, fontweight="bold", loc="left")
ax1.set_xlim(0, 9.0)
ax1.legend(loc="upper right", framealpha=0.94, facecolor="#ffffff", edgecolor="#CBD5E1")
ax1.grid(True, linestyle="--", alpha=0.45)

ax2 = axes[1]
scatter = ax2.scatter(
    bs_events["distance_shore_km"],
    bs_events["avg_speed_knots"],
    c=bs_events["year"],
    cmap="viridis",
    alpha=0.65,
    s=26,
    edgecolors="none",
)
ax2.axvline(5.556, color="#D97706", linestyle="--", linewidth=1.4, label="3-Nautical-Mile Trawl Limit (5.56 km)")
ax2.axhline(3.5, color="#15803D", linestyle=":", linewidth=1.4, label="Low-Speed Working Threshold (3.5 kn)")

cbar = fig.colorbar(scatter, ax=ax2, pad=0.02)
cbar.set_label("Observation Year (2018–2024)", fontsize=10.5, fontweight="bold")
cbar.locator = ticker.MaxNLocator(integer=True)
cbar.update_ticks()

ax2.set_xlabel("Distance from Mainland Coastline (km)", fontweight="bold")
ax2.set_ylabel("Fishing-Event Speed Over Ground (knots)", fontweight="bold")
ax2.set_title("(b) Offshore Distance vs. Active Fishing Speed (GFW AIS Subset)", pad=8, fontweight="bold", loc="left")
ax2.set_xlim(0, 35)
ax2.set_ylim(0, 9.0)
ax2.legend(loc="upper right", framealpha=0.94, facecolor="#ffffff", edgecolor="#CBD5E1")
ax2.grid(True, linestyle="--", alpha=0.45)

fig.savefig(os.path.join(FIGURES_DIR, "figS2_ais_speed_validation.png"), dpi=600)
fig.savefig(os.path.join(FIGURES_DIR, "figS2_ais_speed_validation.pdf"), dpi=600)
plt.close(fig)
print("Rendered Figure S2 (600 DPI) -> figures/figS2_ais_speed_validation.png & .pdf.")
