#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT: 08_generate_publication_figures.py
STUDY:  Decarbonizing Turkish Black Sea Commercial Fisheries Requires Pairing
        Industrial Engine-Power Limits with Capacity Reduction
JOURNAL: Turkish Journal of Maritime and Marine Sciences (DOI: 10.52998/trjmms.2042258)
AUTHOR: Sezgin Tunca, Ph.D. (ORCID: 0000-0001-6840-1466)
PURPOSE: Render all main-text publication figures (Figures 1-7) and Supplementary
         Figure S1 in 600 DPI PNG and vector PDF formats:
           - Figure 1: Integrated Methodological Workflow Diagram
           - Figure 2: Master Study Area Map, GSA 29 Zonation & Fleet Capacity Dashboard
           - Figure 3: Naval Architecture Scaling, Froude Regimes & Holtrop-Mennen Resistance
           - Figure 4: Regional Area 37 Production, GSA 29 Accounting & Economic Context
           - Figure 5: Port-Level Spatial Capacity Distribution & Dual Moran's I Diagnostics
           - Figure 6: Conditional Decarbonization Scenarios, 2023 Fuel Shock & Portfolio Risk
           - Figure 7: Cohort Data Envelopment Analysis (DEA) & Abatement Slack
           - Figure S1: Age Structure, Empirical Moran Scatter & Parametric CO2 Sensitivity
OUTPUTS:
  - figures/fig1_methodological_workflow.png & .pdf
  - figures/fig2_study_area_map.png & .pdf
  - figures/fig3_hydrodynamic_froude_scaling.png & .pdf
  - figures/fig4_comparative_blacksea_dynamics.png & .pdf
  - figures/fig5_turkish_port_bubble_maps.png & .pdf
  - figures/fig6_decarbonisation_tradeoffs.png & .pdf
  - figures/fig7_dea_eco_efficiency_frontiers.png & .pdf
  - figures/figS1_supplementary_diagnostics.png & .pdf
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import seaborn as sns
import geopandas as gpd
from shapely.geometry import box

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TABLES_DIR = os.path.join(BASE_DIR, "tables")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="ticks")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": 11.2,
    "axes.labelsize": 11.8,
    "axes.titlesize": 12.6,
    "xtick.labelsize": 10.2,
    "ytick.labelsize": 10.2,
    "legend.fontsize": 9.6,
    "figure.titlesize": 13.8,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.format": "png",
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.85,
    "xtick.major.width": 0.85,
    "ytick.major.width": 0.85,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.grid": True,
    "axes.axisbelow": True,
})

co_tr = "#DC2626"
col_bgr = "#0284C7"
col_ukr = "#D97706"
col_rou = "#15803D"
col_rus = "#7E22CE"
col_geo = "#DB2777"
col_navy = "#1E3A8A"
col_slate = "#475569"

gear_colors = {
    "Purse Seiner (PS)": "#DC2626",
    "Pelagic/Demersal Trawler (OTB/PTM)": "#0284C7",
    "Beam Trawler (TBB)": "#D97706",
    "Hydraulic Dredger (DRB)": "#7E22CE",
    "Small-Scale / Polyvalent (MIS)": "#15803D",
}

print("=" * 75)
print("GENERATING 600 DPI PUBLICATION FIGURES (FIGURES 1-7 AND FIGURE S1)")
print("=" * 75)

# Load core datasets
fleet_df = pd.read_csv(os.path.join(DATA_DIR, "gfcm_blacksea_fleet.csv"), low_memory=False)
tr_fleet = fleet_df[fleet_df["country"] == "Türkiye"].copy()
hydro_vessels = pd.read_csv(os.path.join(DATA_DIR, "turkish_hydrodynamic_vessel_estimates.csv"), low_memory=False)
hydro_summary = pd.read_csv(os.path.join(TABLES_DIR, "hydrodynamic_regime_summary_turkiye.csv"))
fao_annual = pd.read_csv(os.path.join(DATA_DIR, "fao_blacksea_catches_1950_2024.csv"))
df_sens = pd.read_csv(os.path.join(TABLES_DIR, "fuel_sensitivity_analysis.csv"))
df_depr = pd.read_csv(os.path.join(TABLES_DIR, "depreciation_sensitivity_analysis.csv"))
df_scen = pd.read_csv(os.path.join(TABLES_DIR, "decarbonisation_scenarios_summary.csv"))
df_mc = pd.read_csv(os.path.join(DATA_DIR, "monte_carlo_runs.csv"))
panel_df = pd.read_csv(os.path.join(DATA_DIR, "turkish_basin_panel_data.csv"))
df_ports = pd.read_csv(os.path.join(DATA_DIR, "turkish_geocoded_ports_data.csv"))
df_ports_all34 = pd.read_csv(os.path.join(TABLES_DIR, "turkish_port_level_capacity_all34.csv"))
df_moran_pts = pd.read_csv(os.path.join(DATA_DIR, "moran_scatter_eii_27ports.csv"))

gsa_path = os.path.join(DATA_DIR, "gsas.geojson")
land_path = os.path.join(DATA_DIR, "ne_50m_land.geojson")
countries_path = os.path.join(DATA_DIR, "ne_50m_admin_0_countries.geojson")
bathy_path = os.path.join(DATA_DIR, "ne_10m_bathymetry_K_200.geojson")
rivers_path = os.path.join(DATA_DIR, "ne_50m_rivers_lake_centerlines.geojson")

gsa_gdf = gpd.read_file(gsa_path).to_crs("EPSG:4326") if os.path.exists(gsa_path) else None
land_gdf = gpd.read_file(land_path).to_crs("EPSG:4326") if os.path.exists(land_path) else None
countries_gdf = gpd.read_file(countries_path).to_crs("EPSG:4326") if os.path.exists(countries_path) else None
bathy_gdf = gpd.read_file(bathy_path).to_crs("EPSG:4326") if os.path.exists(bathy_path) else None
rivers_gdf = gpd.read_file(rivers_path).to_crs("EPSG:4326") if os.path.exists(rivers_path) else None


def save_pub_fig(fig, name):
    for ext in ["png", "pdf"]:
        out_path = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        print(f"  [SAVED] figures/{name}.{ext} (600 DPI)")
    plt.close(fig)


PORT_CLEAN_NAMES = {
    "BANDIRMA": "Bandırma*",
    "IGNEADA": "İğneada",
    "ISTANBUL": "İstanbul",
    "SILE": "Şile",
    "KEFKEN": "Kefken",
    "KARASU": "Karasu",
    "KDZ. EREGLI": "Kdz. Ereğli",
    "ZONGULDAK": "Zonguldak",
    "BARTIN": "Bartın",
    "AMASRA": "Amasra",
    "CIDE": "Cide",
    "INEBOLU": "İnebolu",
    "AYANCIK": "Ayancık",
    "SINOP": "Sinop",
    "GERZE": "Gerze",
    "SAMSUN": "Samsun",
    "UNYE": "Ünye",
    "FATSA": "Fatsa",
    "ORDU": "Ordu",
    "GIRESUN": "Giresun",
    "TIREBOLU": "Tirebolu",
    "GORELE": "Görele",
    "VAKFIKEBIR": "Vakfıkebir",
    "TRABZON": "Trabzon",
    "SURMENE": "Sürmene",
    "RIZE": "Rize",
    "PAZAR": "Pazar",
    "HOPA": "Hopa",
}

PORT_OFFSETS_OPTIMAL = {
    "BANDIRMA":    (-0.06,  0.26, "center", "bottom"),
    "IGNEADA":     (-0.12,  0.22, "center", "bottom"),
    "ISTANBUL":    ( 0.14, -0.26, "left",   "top"),
    "SILE":        ( 0.00,  0.24, "center", "bottom"),
    "KEFKEN":      ( 0.00, -0.22, "center", "top"),
    "KARASU":      ( 0.00,  0.26, "center", "bottom"),
    "KDZ. EREGLI": ( 0.00, -0.22, "center", "top"),
    "ZONGULDAK":   (-0.16,  0.25, "right",  "bottom"),
    "BARTIN":      ( 0.02, -0.26, "center", "top"),
    "AMASRA":      ( 0.16,  0.25, "left",   "bottom"),
    "CIDE":        ( 0.06, -0.22, "center", "top"),
    "INEBOLU":     ( 0.00,  0.24, "center", "bottom"),
    "AYANCIK":     (-0.16, -0.22, "center", "top"),
    "SINOP":       ( 0.08,  0.26, "center", "bottom"),
    "GERZE":       ( 0.16, -0.22, "left",   "top"),
    "SAMSUN":      ( 0.00,  0.28, "center", "bottom"),
    "UNYE":        ( 0.00, -0.20, "center", "top"),
    "FATSA":       ( 0.00,  0.22, "center", "bottom"),
    "ORDU":        ( 0.00, -0.52, "center", "top"),
    "GIRESUN":     ( 0.00,  0.64, "center", "bottom"),
    "TIREBOLU":    ( 0.00, -0.20, "center", "top"),
    "GORELE":      ( 0.00,  0.22, "center", "bottom"),
    "VAKFIKEBIR":  (-0.06, -0.52, "center", "top"),
    "TRABZON":     ( 0.00,  0.64, "center", "bottom"),
    "SURMENE":     ( 0.04, -0.22, "center", "top"),
    "RIZE":        ( 0.00,  0.22, "center", "bottom"),
    "PAZAR":       ( 0.12, -0.52, "left",   "top"),
    "HOPA":        (-0.02,  0.26, "right",  "bottom"),
}


# ==============================================================================
# FIGURE 2: MASTER STUDY AREA MAP, FLEET ZONATION & CAPACITY DASHBOARD
# ==============================================================================
print("\n--> Generating Figure 2: Master Study Area Map, Fleet Zonation & Capacity Dashboard...")

fig_w, fig_h = 15.0, 14.0
fig2 = plt.figure(figsize=(fig_w, fig_h), dpi=600)

xlim = (25.5, 42.8)
ylim = (39.8, 47.6)
mean_lat = (ylim[0] + ylim[1]) / 2.0
R_map = ((xlim[1] - xlim[0]) / (ylim[1] - ylim[0])) * np.cos(np.radians(mean_lat))

top_left, top_right = 0.052, 0.978
w_in = (top_right - top_left) * fig_w
h_frac = (w_in / R_map) / fig_h
top_y1 = 0.968
cell_bot = top_y1 - h_frac - 0.008

gs_top = fig2.add_gridspec(1, 1, left=top_left, right=top_right, top=top_y1, bottom=cell_bot)
ax1_map = fig2.add_subplot(gs_top[0])

bot_y1 = cell_bot - 0.048
bot_y0 = 0.046
gs_bot = fig2.add_gridspec(1, 2, left=0.125, right=top_right, top=bot_y1, bottom=bot_y0, width_ratios=[0.84, 1.0], wspace=0.25)
ax1_cap = fig2.add_subplot(gs_bot[0])
ax1_gear = fig2.add_subplot(gs_bot[1])

ax1_map.set_facecolor("#F8FAFC")
bbox_map = box(xlim[0], ylim[0], xlim[1], ylim[1])

if gsa_gdf is not None:
    gsa_gdf[gsa_gdf["f_gsa_lib"] == "GSA 29"].plot(ax=ax1_map, color="#E0F2FE", edgecolor="#0284C7", linewidth=1.4, zorder=2)
    gsa_gdf[gsa_gdf["f_gsa_lib"] == "GSA 28"].plot(ax=ax1_map, color="#F1F5F9", edgecolor="#64748B", linewidth=1.0, linestyle="--", zorder=2)
    gsa_gdf[gsa_gdf["f_gsa_lib"] == "GSA 22"].plot(ax=ax1_map, color="#F1F5F9", edgecolor="#64748B", linewidth=0.8, linestyle="--", zorder=2)
    gsa_gdf[gsa_gdf["f_gsa_lib"] == "GSA 30"].plot(ax=ax1_map, color="#FEF3C7", edgecolor="#D97706", linewidth=1.0, linestyle=":", alpha=0.7, zorder=2)

if bathy_gdf is not None:
    bathy_gdf.clip(bbox_map).plot(ax=ax1_map, color="#38BDF8", linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
if rivers_gdf is not None:
    rivers_gdf.clip(bbox_map).plot(ax=ax1_map, color="#60A5FA", linewidth=1.0, alpha=0.85, zorder=4)
if land_gdf is not None:
    land_gdf.clip(bbox_map).plot(ax=ax1_map, color="#F1F5F9", edgecolor="#94A3B8", linewidth=0.8, zorder=5)
if countries_gdf is not None:
    countries_gdf.clip(bbox_map).plot(ax=ax1_map, color="none", edgecolor="#64748B", linewidth=0.7, linestyle=":", zorder=6)

ax1_map.text(
    34.2, 44.0, "BLACK SEA BASIN (GFCM GSA 29)\n(Area: ~436,402 km² | Mean Depth: 1,253 m)",
    fontsize=12.6, fontweight="bold", color="#0369A1", ha="center", va="center",
    path_effects=[pe.withStroke(linewidth=3.5, foreground="white")], zorder=8,
)
ax1_map.text(27.6, 40.7, "Sea of Marmara\n(GSA 28)*", fontsize=9.0, fontweight="bold", color="#475569",
             ha="center", va="center", path_effects=[pe.withStroke(linewidth=2.5, foreground="white")], zorder=8)
ax1_map.text(26.2, 40.1, "Aegean Sea", fontsize=9.2, fontweight="bold", color="#475569",
             ha="center", va="center", path_effects=[pe.withStroke(linewidth=2.5, foreground="white")], zorder=8)
ax1_map.text(36.8, 46.2, "Sea of Azov", fontsize=9.8, fontweight="bold", color="#B45309",
             ha="center", va="center", path_effects=[pe.withStroke(linewidth=2.5, foreground="white")], zorder=8)
ax1_map.text(31.2, 44.8, "Northwestern Continental Shelf\n(200 m Isobath / Pelagic Spawning Ground)",
             fontsize=9.8, fontstyle="italic", color="#0284C7", ha="center",
             path_effects=[pe.withStroke(linewidth=2.5, foreground="white")], zorder=8)

c_labels = {
    "TÜRKİYE": (35.0, 40.32),
    "BULGARIA": (25.9, 42.8),
    "ROMANIA": (27.2, 44.8),
    "UKRAINE": (32.8, 46.3),
    "GEORGIA": (42.05, 42.55),
    "RUSSIAN FED.": (39.2, 44.4),
}
for cname, (cx, cy) in c_labels.items():
    ax1_map.text(cx, cy, cname, fontsize=10.8, fontweight="bold", color="#1E293B", ha="center", va="center",
                 path_effects=[pe.withStroke(linewidth=3.0, foreground="white")], zorder=8)

ax1_map.text(29.6, 45.2, "Danube Delta", fontsize=9.2, color="#1E40AF", fontstyle="italic",
             path_effects=[pe.withStroke(linewidth=2.2, foreground="white")], zorder=8)
ax1_map.text(32.1, 46.8, "Dnieper River", fontsize=9.2, color="#1E40AF", fontstyle="italic",
             path_effects=[pe.withStroke(linewidth=2.2, foreground="white")], zorder=8)

for _, r in df_ports.iterrows():
    kw = r["Total_kW"]
    bubble_size = np.clip(np.sqrt(kw) * 1.05, 35, 480)
    is_bs = bool(r["Is_BlackSea_Coast"])
    marker_shape = "o" if is_bs else "D"
    fill_col = co_tr if is_bs else "#D97706"
    ax1_map.scatter(
        r["Longitude"], r["Latitude"], s=bubble_size, marker=marker_shape,
        color=fill_col, alpha=0.84, edgecolors="black", linewidth=0.75, zorder=10,
    )
    p_code = r["Port"]
    p_name = PORT_CLEAN_NAMES.get(p_code, p_code.title())
    dx, dy, ha, va = PORT_OFFSETS_OPTIMAL.get(p_code, (0.0, 0.25, "center", "bottom"))
    kw_mw = kw / 1000.0
    ax1_map.annotate(
        f"{p_name}\n({kw_mw:.1f} MW)",
        xy=(r["Longitude"], r["Latitude"]),
        xytext=(r["Longitude"] + dx, r["Latitude"] + dy),
        fontsize=6.5, weight="bold", color="#0F172A", ha=ha, va=va,
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", alpha=0.92, edgecolor="#94A3B8", linewidth=0.4),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.04", color="#64748B", lw=0.4),
        zorder=12,
    )

ax1_map.annotate(
    "East Black Sea Operational Basin\n(Anchovy Overwintering Zone & Purse Seine Focus)",
    xy=(38.8, 41.5), xytext=(36.5, 42.6),
    arrowprops=dict(facecolor=co_tr, arrowstyle="->", lw=1.5, mutation_scale=14),
    fontsize=9.8, fontweight="bold", color=co_tr,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.92, edgecolor=co_tr, lw=1.0),
    zorder=14,
)
ax1_map.annotate(
    "West Black Sea Operational Basin\n(Continental Shelf Trawling & Mixed Gears)",
    xy=(31.2, 41.8), xytext=(28.4, 43.15),
    arrowprops=dict(facecolor=col_navy, arrowstyle="->", lw=1.5, mutation_scale=14),
    fontsize=9.8, fontweight="bold", color=col_navy,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.92, edgecolor=col_navy, lw=1.0),
    zorder=14,
)

for idx, lp in enumerate([10000, 50000, 100000]):
    bsize = np.clip(np.sqrt(lp) * 1.05, 35, 480)
    lx, ly = 26.5 + idx * 1.0, 46.85
    ax1_map.scatter(lx, ly, s=bsize, color="#94A3B8", alpha=0.7, edgecolors="black", linewidth=0.7, zorder=15)
    ax1_map.text(lx, ly - 0.28, f"{lp//1000}k kW\n({lp/1000:g} MW)", fontsize=7.8, ha="center", fontweight="bold", color="#334155",
                 path_effects=[pe.withStroke(linewidth=2.0, foreground="white")], zorder=16)
ax1_map.text(27.5, 47.30, "Port Fleet Engine Power (*Marmara GSA 29 Interface):", fontsize=8.8, fontweight="bold", color="#1E293B", ha="center", zorder=16)

sb_x, sb_y = 30.8, 47.2
sb_len_deg = 200 / 82.0
ax1_map.plot([sb_x, sb_x + sb_len_deg], [sb_y, sb_y], color="black", linewidth=2.5, zorder=15)
ax1_map.plot([sb_x, sb_x], [sb_y - 0.08, sb_y + 0.08], color="black", linewidth=1.5, zorder=15)
ax1_map.plot([sb_x + sb_len_deg, sb_x + sb_len_deg], [sb_y - 0.08, sb_y + 0.08], color="black", linewidth=1.5, zorder=15)
ax1_map.text(sb_x + sb_len_deg / 2.0, sb_y + 0.14, "200 km", fontsize=9.5, fontweight="bold", ha="center", zorder=16,
             path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
ax1_map.annotate("N", xy=(34.0, 47.3), xytext=(34.0, 46.6),
                 arrowprops=dict(facecolor="black", edgecolor="black", width=1.5, headwidth=6),
                 ha="center", va="center", fontsize=10.5, fontweight="bold", zorder=16)

ax_inset = ax1_map.inset_axes([0.75, 0.63, 0.23, 0.34])
ax_inset.set_facecolor("#F0F9FF")
if countries_gdf is not None:
    countries_gdf.plot(ax=ax_inset, color="#E2E8F0", edgecolor="#CBD5E1", linewidth=0.4)
    if gsa_gdf is not None:
        gsa_gdf[gsa_gdf["f_gsa_lib"] == "GSA 29"].plot(ax=ax_inset, color="#0284C7", edgecolor="none", zorder=5)
    countries_gdf[countries_gdf["NAME"].isin(["Turkey", "Türkiye"])].plot(ax=ax_inset, color=co_tr, edgecolor="black", linewidth=0.5, zorder=6)
ax_inset.set_xlim(-6, 45)
ax_inset.set_ylim(30, 48)
ax_inset.set_xticks([])
ax_inset.set_yticks([])
for spine in ax_inset.spines.values():
    spine.set_edgecolor("#475569")
    spine.set_linewidth(0.8)
ax_inset.text(35, 38.8, "Türkiye\n(Study Area)", fontsize=8.2, fontweight="bold", color=co_tr, ha="center",
              path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
ax_inset.text(34, 43.5, "Black Sea", fontsize=7.8, fontweight="bold", color="white", ha="center",
              path_effects=[pe.withStroke(linewidth=2.0, foreground="#0284C7")])
ax_inset.set_title("Regional Overview", fontsize=8.8, fontweight="bold", color="#1E293B", pad=2)

ax1_map.set_xlim(xlim[0], xlim[1])
ax1_map.set_ylim(ylim[0], ylim[1])
ax1_map.set_xlabel("Longitude (°E)", fontweight="bold", fontsize=11.5)
ax1_map.set_ylabel("Latitude (°N)", fontweight="bold", fontsize=11.5)
ax1_map.set_title("(a) Black Sea Basin, 27 Coastal Ports + Bandırma GSA 29 Interface & Operational Zones", loc="left", fontweight="bold", fontsize=12.6)

# Panel B: 21 DEA Operational Segments (n = 4,772)
arch_df = pd.read_csv(os.path.join(TABLES_DIR, "turkish_dea_gear_segment_results.csv")).sort_values(by="Total_kW", ascending=True)
y_indices = np.arange(len(arch_df))
bar_h = 0.36
ax1_cap.barh(y_indices + bar_h / 2, arch_df["Total_kW"] / 1000.0, height=bar_h, color=co_tr, edgecolor="black", linewidth=0.5, label="Installed Engine Power (MW)")
ax1_cap.barh(y_indices - bar_h / 2, arch_df["Total_GT"] / 1000.0, height=bar_h, color=col_navy, edgecolor="black", linewidth=0.5, label="Gross Tonnage (kGT)")

seg_clean = [
    s.replace("Pelagic/Demersal Trawler (OTB/PTM)", "Trawler")
     .replace("Small-Scale / Polyvalent (MIS)", "Small-Scale")
     .replace("Purse Seiner (PS)", "Purse Seine")
     .replace("Beam Trawler (TBB)", "Beam Trawl")
     .replace("very_large (>=24m)", "≥24m")
     .replace("large (18-24m)", "18–24m")
     .replace("medium (12-18m)", "12–18m")
     .replace("small (6-12m)", "6–12m")
    for s in arch_df["Segment"]
]
ax1_cap.set_yticks(y_indices)
ax1_cap.set_yticklabels(seg_clean, fontweight="bold", fontsize=9.2)
ax1_cap.set_xlabel("Cohort Capacity (MW or Thousand GT)", fontweight="bold", fontsize=10.8)
ax1_cap.set_title("(b) Capacity Across 21 Black Sea Cohorts (n = 4,772; ≥6 m)", loc="left", fontweight="bold", fontsize=11.8)
ax1_cap.set_xlim(0, max(arch_df["Total_kW"] / 1000.0) * 1.48)
ax1_cap.legend(loc="lower right", framealpha=0.92, fontsize=9.2)

for idx_b, r_b in arch_df.reset_index().iterrows():
    kw_mw = r_b["Total_kW"] / 1000.0
    gt_kgt = r_b["Total_GT"] / 1000.0
    v_cnt = int(r_b["Vessel_Count"])
    ax1_cap.text(max(kw_mw, gt_kgt) + 2.5, idx_b, f"{kw_mw:.1f} MW | {gt_kgt:.1f} kGT ({v_cnt:,} v.)",
                 va="center", fontsize=7.9, fontweight="bold", color="#1E293B")

# Panel C: Top 12 Registration Ports (from full 34-port audit)
df_ports_top = df_ports_all34.head(12).sort_values(by="Total_kW", ascending=True)
y_p_indices = np.arange(len(df_ports_top))
bar_ph = 0.36
ax1_gear.barh(y_p_indices + bar_ph / 2, df_ports_top["Total_kW"] / 1000.0, height=bar_ph, color="#E11D48", edgecolor="black", linewidth=0.5, label="Installed Engine Power (MW)")
ax1_gear.barh(y_p_indices - bar_ph / 2, df_ports_top["Vessel_Count"] / 10.0, height=bar_ph, color="#475569", edgecolor="black", linewidth=0.5, label="Registered Vessels (×10)")

port_names_clean = [
    PORT_CLEAN_NAMES.get(p, p.title())
    for p in df_ports_top["Port"]
]
ax1_gear.set_yticks(y_p_indices)
ax1_gear.set_yticklabels(port_names_clean, fontweight="bold", fontsize=9.5)
ax1_gear.set_xlabel("Port Capacity Scale (MW engine power | Tens of vessels)", fontweight="bold", fontsize=10.8)
ax1_gear.set_title("(c) Top 12 Registration Ports (of 34 Total; *Marmara Interface)", loc="left", fontweight="bold", fontsize=11.8)
ax1_gear.set_xlim(0, max(df_ports_top["Total_kW"] / 1000.0) * 1.36)
ax1_gear.legend(loc="lower right", framealpha=0.92, fontsize=9.2)

for idx_p, r_p in df_ports_top.reset_index().iterrows():
    pkw = r_p["Total_kW"] / 1000.0
    pvc = int(r_p["Vessel_Count"])
    ax1_gear.text(pkw + 2.2, idx_p + bar_ph / 2, f"{pkw:.1f} MW", va="center", fontsize=8.3, fontweight="bold", color="#E11D48")
    ax1_gear.text((pvc / 10.0) + 2.2, idx_p - bar_ph / 2, f"{pvc:,} vessels", va="center", fontsize=8.3, fontweight="bold", color="#475569")

save_pub_fig(fig2, "fig2_study_area_map")


# ==============================================================================
# FIGURE 3: NAVAL ARCHITECTURE, EMPIRICAL FROUDE REGIMES & HOLTROP-MENNEN DRAG
# ==============================================================================
print("\n--> Generating Figure 3: Naval Architecture Scaling and Holtrop-Mennen Resistance...")

fig3, axes3 = plt.subplots(2, 2, figsize=(14.5, 10.8), dpi=600)

# Panel A: Empirical ln(EII) vs ln(LOA) across all 5 gears (N = 6,584)
ax_3a = axes3[0, 0]
hv_valid = hydro_vessels[(hydro_vessels["eii"] > 0) & (hydro_vessels["loa_m"] > 0)].copy()
hv_valid["ln_eii"] = np.log(hv_valid["eii"])
hv_valid["ln_loa"] = np.log(hv_valid["loa_m"])

gear_short_labels = {
    "Purse Seiner (PS)": "Purse Seiners",
    "Pelagic/Demersal Trawler (OTB/PTM)": "Trawlers (OTB/PTM)",
    "Beam Trawler (TBB)": "Beam Trawlers (TBB)",
    "Hydraulic Dredger (DRB)": "Hydraulic Dredgers (DRB)",
    "Small-Scale / Polyvalent (MIS)": "Small-Scale Fleet (MIS)",
}

for gear, color in gear_colors.items():
    sub = hv_valid[hv_valid["gear_detailed"] == gear]
    if len(sub) > 0:
        ax_3a.scatter(
            sub["ln_loa"], sub["ln_eii"], alpha=0.34 if len(sub) > 100 else 0.75,
            s=16 if len(sub) > 100 else 28, color=color,
            label=f"{gear_short_labels[gear]} (n = {len(sub):,})",
        )
        if len(sub) >= 20:
            sns.regplot(data=sub, x="ln_loa", y="ln_eii", ax=ax_3a, scatter=False, color=color, line_kws={"linewidth": 1.8})

ax_3a.axhline(np.log(6.0), color="#1E293B", linestyle="--", linewidth=1.1, label=r"Purse Seine Upper Benchmark (6.0 kW GT$^{-1}$)")
ax_3a.axhline(np.log(10.0), color="#DC2626", linestyle=":", linewidth=1.2, label=r"Trawler Upper Benchmark (10.0 kW GT$^{-1}$)")

ax_3a.set_title(r"(a) Within-Gear Scaling: $\ln(\mathrm{EII})$ vs. $\ln(\mathrm{LOA})$ (N = 6,584)", fontsize=11.8, weight="bold", loc="left")
ax_3a.set_xlabel(r"$\ln(\mathrm{Length\ Overall\ [m]})$", fontsize=10.8, weight="bold")
ax_3a.set_ylabel(r"$\ln(\mathrm{Engine\ Intensity\ Index\ [kW\ GT^{-1}]})$", fontsize=10.8, weight="bold")
ax_3a.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.4)

# Panel B: Vessel-Level Transit Froude Number Distributions
ax_3b = axes3[0, 1]
ax_3b.axvspan(0.18, 0.34, color="#DCFCE7", alpha=0.45, label=r"Displacement Transit ($Fn < 0.34$)")
ax_3b.axvspan(0.34, 0.62, color="#FEE2E2", alpha=0.45, label=r"Wave-Dominated Transit ($Fn \geq 0.34$)")

for gear, color, lbl in [
    ("Purse Seiner (PS)", "#DC2626", "Purse Seiners (11.5 kn, mean Fn = 0.348)"),
    ("Pelagic/Demersal Trawler (OTB/PTM)", "#0284C7", "Trawlers (9.5 kn, mean Fn = 0.369)"),
    ("Beam Trawler (TBB)", "#D97706", "Beam Trawlers (7.0 kn, mean Fn = 0.444)"),
    ("Small-Scale / Polyvalent (MIS)", "#15803D", "Small-Scale (6.5 kn, mean Fn = 0.421)"),
]:
    fn_vals = hv_valid.loc[hv_valid["gear_detailed"] == gear, "Froude_Number_Fn"].dropna()
    sns.kdeplot(fn_vals, ax=ax_3b, color=color, fill=True, alpha=0.22, linewidth=1.9, bw_adjust=1.1, clip=(0.18, 0.62), label=lbl)

ax_3b.axvline(0.34, color="#B45309", linestyle="--", linewidth=1.4, label=r"50% Wave-Drag Crossover ($Fn \approx 0.34$)")
ax_3b.annotate(
    "Holtrop–Mennen Wave Drag\nExceeds 50% Above Fn ≈ 0.34",
    xy=(0.41, 7.2), xytext=(0.415, 11.2),
    arrowprops=dict(facecolor="#B91C1C", edgecolor="#B91C1C", arrowstyle="->", lw=1.2),
    fontsize=8.8, fontweight="bold", color="#B91C1C",
    bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.94, edgecolor="#B91C1C"),
)

ax_3b.set_title(r"(b) Empirical Transit Froude Distributions ($Fn = v_{\mathrm{tr}} / \sqrt{g \cdot \mathrm{LOA}}$)", fontsize=11.8, weight="bold", loc="left")
ax_3b.set_xlabel("Free-Running Transit Froude Number ($Fn$)", fontsize=10.8, weight="bold")
ax_3b.set_ylabel("Empirical Kernel Density (6,584 Vessels)", fontsize=10.8, weight="bold")
ax_3b.set_xlim(0.18, 0.62)
ax_3b.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.3)

# Panel C: Gear-Specific Benchmark Exceedance vs Uniform >10 kW/GT Threshold
ax_3c = axes3[1, 0]
order_gears = [
    "Purse Seiner (PS)",
    "Pelagic/Demersal Trawler (OTB/PTM)",
    "Beam Trawler (TBB)",
    "Small-Scale / Polyvalent (MIS)",
    "Hydraulic Dredger (DRB)",
]
hs_ord = hydro_summary.set_index("gear_detailed").loc[order_gears].reset_index()

x_pos = np.arange(len(hs_ord))
w_c = 0.36
b_uni = ax_3c.bar(
    x_pos - w_c / 2, hs_ord["Pct_Overpowered"], width=w_c,
    color="#64748B", edgecolor="#1E293B", linewidth=0.6, alpha=0.88,
    label=r"Uniform Threshold ($\mathrm{EII} > 10\ \mathrm{kW\ GT}^{-1}$)",
)
b_gear = ax_3c.bar(
    x_pos + w_c / 2, hs_ord["Pct_Exceeding_Gear_Benchmark"], width=w_c,
    color="#DC2626", edgecolor="#1E293B", linewidth=0.6, alpha=0.88,
    label="Gear-Specific Upper Benchmark (PS>6, OTB/DRB>10, TBB/MIS>12)",
)

for bar in b_uni:
    h = bar.get_height()
    ax_3c.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.4, weight="bold", color="#334155")
for bar in b_gear:
    h = bar.get_height()
    ax_3c.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.4, weight="bold", color="#991B1B")

ax_3c.set_xticks(x_pos)
ax_3c.set_xticklabels([
    "Purse Seiners\n(n = 208)",
    "Trawlers\n(n = 549)",
    "Beam Trawlers\n(n = 548)",
    "Small-Scale\n(n = 5,259)",
    "Dredgers\n(n = 20)",
], fontsize=9.0, weight="bold")
ax_3c.set_title("(c) High Engine Intensity Share: Uniform vs. Gear-Specific Benchmarks", fontsize=11.8, weight="bold", loc="left")
ax_3c.set_ylabel("Share of Segment Vessels (%)", fontsize=10.8, weight="bold")
ax_3c.set_ylim(0, 72)
ax_3c.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.4)

# Panel D: Holtrop-Mennen + ITTC-1957 Wave Resistance Share vs LOA
ax_3d = axes3[1, 1]
for gear, color in gear_colors.items():
    sub = hv_valid[hv_valid["gear_detailed"] == gear].sort_values("loa_m")
    if len(sub) > 0:
        ax_3d.scatter(
            sub["loa_m"], sub["Wave_Resistance_Share"] * 100.0,
            s=14 if len(sub) > 100 else 28, alpha=0.35 if len(sub) > 100 else 0.80,
            color=color, label=f"{gear_short_labels[gear]} Transit",
        )

# Overlay active working-speed (towing/setting) wave share (<1.3%)
hv_work = hv_valid.sort_values("loa_m")
ax_3d.scatter(
    hv_work["loa_m"][::8], hv_work["Working_Wave_Share"][::8] * 100.0,
    s=10, alpha=0.45, color="#0F172A", marker="x", label="Active Towing / Purse-Setting (2.0–3.0 kn: <1.3%)",
)
ax_3d.axhline(50.0, color="#D97706", linestyle="--", linewidth=1.2, label="Wave-Making Majority Threshold (50%)")

ax_3d.set_title(r"(d) Semi-Empirical Holtrop–Mennen Wave Drag Share ($R_W / R_T$)", fontsize=11.8, weight="bold", loc="left")
ax_3d.set_xlabel("Vessel Length Overall, LOA (m)", fontsize=10.8, weight="bold")
ax_3d.set_ylabel("Theoretical Wave-Making Share of Resistance (%)", fontsize=10.8, weight="bold")
ax_3d.set_xlim(3.5, 64.0)
ax_3d.set_ylim(-3, 96)
ax_3d.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.1, ncol=2)

plt.tight_layout(pad=1.8)
save_pub_fig(fig3, "fig3_hydrodynamic_froude_scaling")


# ==============================================================================
# FIGURE 4: COMPARATIVE BLACK SEA FISHERIES PRODUCTION & ECONOMIC BENCHMARKS
# ==============================================================================
print("\n--> Generating Figure 4: Comparative Black Sea Fisheries Production & Economic Benchmarks...")

fig4, axes4 = plt.subplots(2, 2, figsize=(15.0, 11.0), dpi=600)

ax_a = axes4[0, 0]
fao_pivot = fao_annual.pivot(index="Year", columns="Country", values="Catch_t").fillna(0.0) / 1000.0
ax_a.axvspan(1989.0, 1992.5, color="#FEE2E2", alpha=0.55)
ax_a.axvspan(2021.0, 2023.5, color="#FEF3C7", alpha=0.55)
ax_a.text(1991.5, 685, "Anchovy Collapse &\nSoviet Dissolution", fontsize=8.4, fontweight="bold", color="#B91C1C", ha="center", va="top",
          bbox=dict(boxstyle="square,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"), zorder=8)
ax_a.text(2022.25, 685, "Diesel Price\nSurge", fontsize=8.4, fontweight="bold", color="#B45309", ha="center", va="top",
          bbox=dict(boxstyle="square,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"), zorder=8)

ax_a.plot(fao_pivot.index, fao_pivot["Türkiye"], label="Türkiye (FAO Area 37)", color=co_tr, linewidth=2.6, zorder=6)
if "Soviet Union [former]" in fao_pivot.columns:
    ax_a.plot(fao_pivot.index, fao_pivot["Soviet Union [former]"], label="Former Soviet Union", color=col_rus, linewidth=1.8, linestyle="--", alpha=0.85, zorder=5)
if "Ukraine" in fao_pivot.columns:
    ax_a.plot(fao_pivot.index, fao_pivot["Ukraine"], label="Ukraine", color=col_ukr, linewidth=1.6, alpha=0.85, zorder=5)
if "Russian Federation" in fao_pivot.columns:
    ax_a.plot(fao_pivot.index, fao_pivot["Russian Federation"], label="Russian Federation", color="#7C3AED", linewidth=1.5, alpha=0.85, zorder=5)
if "Bulgaria" in fao_pivot.columns:
    ax_a.plot(fao_pivot.index, fao_pivot["Bulgaria"], label="Bulgaria", color=col_bgr, linewidth=1.4, alpha=0.85, zorder=5)
if "Romania" in fao_pivot.columns:
    ax_a.plot(fao_pivot.index, fao_pivot["Romania"], label="Romania", color=col_rou, linewidth=1.4, alpha=0.85, zorder=5)
if "Georgia" in fao_pivot.columns:
    ax_a.plot(fao_pivot.index, fao_pivot["Georgia"], label="Georgia", color=col_geo, linewidth=1.3, alpha=0.85, zorder=5)

ax_a.annotate("Türkiye Area 37 Peak\n(1988: 621.4 kt)", xy=(1988, 621.4), xytext=(1979.0, 628),
              arrowprops=dict(facecolor=co_tr, edgecolor=co_tr, arrowstyle="->", lw=1.2),
              fontsize=8.6, fontweight="bold", color=co_tr, ha="center",
              bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.95, edgecolor=co_tr, lw=0.8), zorder=10)
ax_a.annotate("Soviet Peak\n(1982: 522.2 kt)", xy=(1982, 522.2), xytext=(1972.5, 535),
              arrowprops=dict(facecolor=col_rus, edgecolor=col_rus, arrowstyle="->", lw=1.2),
              fontsize=8.6, fontweight="bold", color=col_rus, ha="center",
              bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.95, edgecolor=col_rus, lw=0.8), zorder=10)

ax_a.set_title("(a) Historical FAO Area 37 Marine Capture Landings by Country (1950–2024)", fontsize=12.0, weight="bold", loc="left")
ax_a.set_xlabel("Year", fontsize=11.2, weight="bold")
ax_a.set_ylabel("Capture Landings (Thousand Tonnes)", fontsize=11.2, weight="bold")
ax_a.set_xlim(1950, 2024)
ax_a.set_ylim(-10, 720)
ax_a.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
ax_a.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.5, framealpha=0.95)

# Panel B: 2015-2024 FAO Area 37 vs TÜİK GSA 29 note
ax_b = axes4[0, 1]
fao_recent = fao_annual[(fao_annual["Year"] >= 2015) & (fao_annual["Country"] != "Soviet Union [former]")].groupby("Country")["Catch_t"].mean().sort_values(ascending=True) / 1000.0
colors_b = [co_tr if c == "Türkiye" else (col_ukr if c == "Ukraine" else (col_rus if "Russia" in c else (col_geo if c == "Georgia" else (col_bgr if c == "Bulgaria" else col_rou)))) for c in fao_recent.index]
bars_b = ax_b.barh(fao_recent.index, fao_recent.values, color=colors_b, edgecolor="#1E293B", linewidth=0.6, alpha=0.86)

for bar in bars_b:
    w = bar.get_width()
    ax_b.text(w + 3.5, bar.get_y() + bar.get_height() / 2.0, f"{w:.1f} kt ({w/fao_recent.sum()*100:.1f}%)", va="center", fontsize=9.8, weight="bold")

# Overlay dashed marker for Türkiye GSA 29 Black Sea-only landings (256.74 kt)
tr_idx = list(fao_recent.index).index("Türkiye")
ax_b.plot([256.74, 256.74], [tr_idx - 0.42, tr_idx + 0.42], color="#0F172A", linestyle="--", linewidth=2.0, zorder=8)
ax_b.annotate(
    "TÜİK GSA 29 Black Sea Only:\n256.7 kt/yr (vs. 340.8 kt All Turkish Seas)",
    xy=(256.74, tr_idx - 0.15), xytext=(155.0, tr_idx - 1.55),
    arrowprops=dict(facecolor="#0F172A", arrowstyle="->", lw=1.2),
    fontsize=8.6, fontweight="bold", color="#0F172A",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", alpha=0.95, edgecolor="#64748B"),
)

ax_b.set_title("(b) Mean Annual FAO Area 37 Landings & GSA 29 Accounting (2015–2024)", fontsize=12.0, weight="bold", loc="left")
ax_b.set_xlabel("Mean Annual Landings (Thousand Tonnes / Year; 6-Country Sum = 514.0 kt)", fontsize=10.8, weight="bold")
ax_b.set_xlim(0, max(fao_recent.values) * 1.28)

# Panel C: Turkish Black Sea Landed Harvest Value & National Trade Inset
ax_c = axes4[1, 0]
bs_panel = panel_df[panel_df["Region"].isin(["East_Black_Sea", "West_Black_Sea"])]
val_usd = bs_panel.pivot(index="Year", columns="Region", values="Total_Value_USD").fillna(0.0) / 1e6
years = val_usd.index
ax_c.bar(years, val_usd["East_Black_Sea"], width=0.65, label="Eastern Black Sea", color=co_tr, alpha=0.88, edgecolor="#1E293B", linewidth=0.5)
ax_c.bar(years, val_usd["West_Black_Sea"], bottom=val_usd["East_Black_Sea"], width=0.65, label="Western Black Sea", color="#0284C7", alpha=0.88, edgecolor="#1E293B", linewidth=0.5)
ax_c.plot(years, val_usd["East_Black_Sea"] + val_usd["West_Black_Sea"], color="#0F172A", linewidth=1.8, marker="o", markersize=3.5, label="Total Turkish Black Sea")

ax_c.set_ylabel("Landed Harvest Value (Million USD)", fontsize=11.2, weight="bold", color="#1E293B")
ax_c.set_xlabel("Year", fontsize=11.2, weight="bold")
ax_c.set_xlim(1999.2, 2024.8)
ax_c.set_ylim(0, 1100)
ax_c.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${int(x):,d}M"))
ax_c.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9.2)

ax_c_inset = ax_c.inset_axes([0.45, 0.50, 0.52, 0.43])
ax_c_inset.set_facecolor("#F8FAFC")
trade_data = [
    ("Georgia", 30.6, 48.2),
    ("Romania", 32.6, 385.8),
    ("Ukraine", 41.3, 735.4),
    ("Bulgaria", 81.9, 135.9),
    ("Türkiye", 1243.8, 562.9),
    ("Russian Fed.", 5048.3, 2282.5),
]
c_names = [d[0] for d in trade_data]
exp_vals = [d[1] for d in trade_data]
imp_vals = [d[2] for d in trade_data]
y_pos_t = np.arange(len(c_names))
bar_ht = 0.34
ax_c_inset.barh(y_pos_t + bar_ht / 2, exp_vals, height=bar_ht, color=[co_tr if c == "Türkiye" else "#2563EB" for c in c_names], edgecolor="#1E293B", linewidth=0.5)
ax_c_inset.barh(y_pos_t - bar_ht / 2, imp_vals, height=bar_ht, color=["#EA580C" if c == "Türkiye" else "#64748B" for c in c_names], edgecolor="#1E293B", linewidth=0.5, hatch="//")
ax_c_inset.set_yticks(y_pos_t)
ax_c_inset.set_yticklabels(c_names, fontsize=8.0, weight="bold")
ax_c_inset.set_xscale("log")
ax_c_inset.set_xlim(10, 45000)
ax_c_inset.tick_params(axis="x", which="major", labelsize=7.8)
ax_c_inset.set_title("National Aquatic Product Trade: Exports vs. Imports\n(Mean 2015–2024, Million USD)", fontsize=8.2, weight="bold", pad=3)
for i, (ev, iv) in enumerate(zip(exp_vals, imp_vals)):
    ax_c_inset.text(ev * 1.18, i + bar_ht / 2, f"${ev:,.0f}M", va="center", fontsize=7.0, weight="bold", color="#1E40AF" if c_names[i] != "Türkiye" else "#991B1B")
    ax_c_inset.text(iv * 1.18, i - bar_ht / 2, f"${iv:,.0f}M", va="center", fontsize=7.0, weight="bold", color="#334155" if c_names[i] != "Türkiye" else "#C2410C")
ax_c_inset.legend(
    handles=[
        mpatches.Patch(facecolor="#2563EB", edgecolor="#1E293B", label="Exports"),
        mpatches.Patch(facecolor="#64748B", edgecolor="#1E293B", hatch="//", label="Imports"),
    ],
    loc="lower right", fontsize=7.8, framealpha=0.9, edgecolor="#CBD5E1",
)
ax_c.set_title("(c) Turkish Black Sea Landed Value (TÜİK) & National Seafood Trade", fontsize=12.0, weight="bold", loc="left")

# Panel D: Dominant Species Landings
ax_d = axes4[1, 1]
sp_summary = pd.read_csv(os.path.join(TABLES_DIR, "fao_blacksea_species_summary.csv")).head(8)
x_sp = np.arange(len(sp_summary))
ax_d.bar(x_sp - 0.18, sp_summary["Turkiye_Catch_kt"], 0.36, label="Türkiye", color=co_tr, edgecolor="#1E293B", linewidth=0.6, alpha=0.86)
ax_d.bar(x_sp + 0.18, sp_summary["Total_Catch_kt"] - sp_summary["Turkiye_Catch_kt"], 0.36, label="Other Black Sea Countries", color="#64748B", edgecolor="#1E293B", linewidth=0.6, alpha=0.86)

clean_species_names = [
    s.replace(" NEI", "").replace("NEI", "").replace("(=Sardine)", " (Sardine)")
     .replace("Black and Caspian Sea sprat", "Black/Caspian Sprat").strip().title()
    for s in sp_summary["Species_Common"]
]
ax_d.set_xticks(x_sp)
ax_d.set_xticklabels(clean_species_names, rotation=22, ha="right", fontsize=9.8, weight="bold")
ax_d.set_title("(d) Dominant Commercial Species Landings in Area 37 (1950–2024)", fontsize=12.0, weight="bold", loc="left")
ax_d.set_ylabel("Cumulative Landings (Thousand Tonnes)", fontsize=11.2, weight="bold")
ax_d.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,d}"))
ax_d.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9.8)

plt.tight_layout(pad=1.0)
save_pub_fig(fig4, "fig4_comparative_blacksea_dynamics")


# ==============================================================================
# FIGURE 5: TURKISH BLACK SEA PORT BUBBLE MAP (4-PANEL, 27 BLACK SEA + BANDIRMA*)
# ==============================================================================
print("\n--> Generating Figure 5: Turkish Black Sea Port Bubble Map (4-Panel)...")

fig5, axes5 = plt.subplots(2, 2, figsize=(15.5, 7.9), dpi=600)
bbox_tr_ports = box(27.0, 40.0, 42.2, 43.6)

land_crop_p = land_gdf.clip(bbox_tr_ports) if land_gdf is not None else None
countries_crop_p = countries_gdf.clip(bbox_tr_ports) if countries_gdf is not None else None
bathy_crop_p = bathy_gdf.clip(bbox_tr_ports) if bathy_gdf is not None else None
rivers_crop_p = rivers_gdf.clip(bbox_tr_ports) if rivers_gdf is not None else None

panel_configs = [
    ("Vessel_Count", "(a) Registered Commercial Fishing Vessels", "Purples", 7.5, axes5[0, 0]),
    ("Total_GT", "(b) Aggregate Fleet Gross Tonnage (GT)", "Blues", 2.2, axes5[0, 1]),
    ("Total_kW", "(c) Total Installed Main Engine Power (kW; Moran's I = -0.069, p = 0.47)", "Reds", 0.85, axes5[1, 0]),
    ("Median_EII", "(d) Median Engine Intensity (kW / GT; Moran's I = +0.641, p = 0.0001)", "YlOrRd", 9.5, axes5[1, 1]),
]

for p_idx, (col_name, title, cmap_name, scale_factor, ax_p) in enumerate(panel_configs):
    r_idx = p_idx // 2
    ax_p.set_facecolor("#E0F2FE")
    if bathy_crop_p is not None and len(bathy_crop_p) > 0:
        bathy_crop_p.plot(ax=ax_p, color="#38BDF8", linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    if rivers_crop_p is not None and len(rivers_crop_p) > 0:
        rivers_crop_p.plot(ax=ax_p, color="#60A5FA", linewidth=0.8, alpha=0.8, zorder=3)
    if land_crop_p is not None and len(land_crop_p) > 0:
        land_crop_p.plot(ax=ax_p, color="#F1F5F9", edgecolor="#94A3B8", linewidth=0.7, zorder=4)
    if countries_crop_p is not None and len(countries_crop_p) > 0:
        countries_crop_p.plot(ax=ax_p, color="none", edgecolor="#CBD5E1", linewidth=0.6, linestyle="--", zorder=4)

    ax_p.text(34.5, 43.1, "BLACK SEA", fontsize=10.0, fontweight="bold", color="#0284C7", alpha=0.6, ha="center",
              path_effects=[pe.withStroke(linewidth=2.2, foreground="white")], zorder=5)
    ax_p.text(35.5, 40.4, "T Ü R K İ Y E", fontsize=12.0, fontweight="bold", color="#64748B", alpha=0.5, ha="center",
              path_effects=[pe.withStroke(linewidth=2.5, foreground="white")], zorder=5)

    sizes = np.clip(df_ports["Median_EII"] * scale_factor, 35, 260) if col_name == "Median_EII" else np.clip(np.sqrt(df_ports[col_name]) * scale_factor, 30, 350)
    sc_p = ax_p.scatter(
        df_ports["Longitude"], df_ports["Latitude"],
        s=sizes, c=df_ports[col_name], cmap=cmap_name,
        edgecolor="#0F172A", linewidth=0.75, alpha=0.88, zorder=7,
    )

    for _, r in df_ports.iterrows():
        p_code = r["Port"]
        p_name = PORT_CLEAN_NAMES.get(p_code, p_code.title())
        dx, dy, ha, va = PORT_OFFSETS_OPTIMAL.get(p_code, (0.0, 0.22, "center", "bottom"))
        if col_name == "Vessel_Count":
            val_txt = f"{int(r['Vessel_Count']):,d}"
        elif col_name == "Total_GT":
            val_txt = f"{r['Total_GT']:,.0f} GT"
        elif col_name == "Total_kW":
            val_txt = f"{r['Total_kW']/1000.0:.1f} MW"
        else:
            val_txt = f"{r['Median_EII']:.1f}"

        ax_p.annotate(
            f"{p_name}\n({val_txt})",
            xy=(r["Longitude"], r["Latitude"]),
            xytext=(r["Longitude"] + dx, r["Latitude"] + dy),
            fontsize=6.3, weight="bold", color="#0F172A", ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.11", facecolor="white", alpha=0.92, edgecolor="#94A3B8", linewidth=0.4),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.03", color="#64748B", lw=0.4),
            zorder=9,
        )

    if p_idx in [0, 2]:
        sb_x, sb_y = 27.8, 43.15
        sb_len = 150 / 82.0
        ax_p.plot([sb_x, sb_x + sb_len], [sb_y, sb_y], color="black", linewidth=1.8, zorder=10)
        ax_p.plot([sb_x, sb_x], [sb_y - 0.05, sb_y + 0.05], color="black", linewidth=1.2, zorder=10)
        ax_p.plot([sb_x + sb_len, sb_x + sb_len], [sb_y - 0.05, sb_y + 0.05], color="black", linewidth=1.2, zorder=10)
        ax_p.text(sb_x + sb_len / 2.0, sb_y + 0.08, "150 km", fontsize=8.0, fontweight="bold", ha="center", zorder=10,
                  path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
        ax_p.annotate("N", xy=(41.3, 43.2), xytext=(41.3, 42.75),
                      arrowprops=dict(facecolor="black", edgecolor="black", width=1.0, headwidth=4.5),
                      ha="center", va="center", fontsize=8.8, fontweight="bold", zorder=10)

    ax_p.set_xlim(27.2, 42.0)
    ax_p.set_ylim(40.1, 43.5)
    ax_p.set_title(title, fontsize=11.0, weight="bold", loc="left", pad=4)
    ax_p.set_ylabel("Latitude (°N)", fontsize=9.8, weight="bold")

    if r_idx == 0:
        ax_p.tick_params(labelbottom=False)
        cbar_pad = 0.05
    else:
        ax_p.set_xlabel("Longitude (°E) [*Bandırma: Marmara GSA 29 Interface]", fontsize=9.6, weight="bold", labelpad=4)
        cbar_pad = 0.16

    cbar_p = plt.colorbar(sc_p, ax=ax_p, orientation="horizontal", fraction=0.045, pad=cbar_pad, aspect=32, shrink=0.88)
    cbar_label = title.split(") ")[1].split(";")[0] + (")" if "(" in title.split(") ")[1].split(";")[0] and not title.split(") ")[1].split(";")[0].endswith(")") else "")
    cbar_p.set_label(cbar_label, fontsize=9.6, weight="bold")
    if col_name != "Median_EII":
        cbar_p.ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

plt.subplots_adjust(left=0.035, right=0.985, top=0.965, bottom=0.075, hspace=0.30, wspace=0.08)
save_pub_fig(fig5, "fig5_turkish_port_bubble_maps")


# ==============================================================================
# FIGURE 7: DATA ENVELOPMENT ANALYSIS (DEA) ECO-EFFICIENCY & REALISTIC CO2 SLACK
# ==============================================================================
print("\n--> Generating Figure 7: Data Envelopment Analysis (DEA) Eco-Efficiency Frontier...")

df_gear_dea = pd.read_csv(os.path.join(TABLES_DIR, "turkish_dea_gear_segment_results.csv"))
clean_labels_b = []
for _, r in df_gear_dea.iterrows():
    g = r["Segment"]
    basin = "East" if "East" in g else "West"
    if "Purse Seiner" in g:
        g = "Purse Seine"
    elif "Trawler (OTB/PTM)" in g:
        g = "Trawler"
    elif "Beam Trawler" in g:
        g = "Beam Trawl"
    elif "Small-Scale" in g:
        g = "Small-Scale"
    loa_part = r["Size_Class"].split("(")[-1].rstrip(")").replace(">=", "≥").replace("-", "–")
    clean_labels_b.append(f"{basin} {g} ({loa_part})")
df_gear_dea["Clean_Label"] = clean_labels_b

fig6 = plt.figure(figsize=(16.0, 13.5), dpi=600)
gs6 = gridspec.GridSpec(2, 2, height_ratios=[1.16, 1.0], hspace=0.34, wspace=0.30, top=0.95, bottom=0.06, left=0.06, right=0.98)

# Panel A: Power vs Landings colored by Eco-Efficiency
ax_6a = fig6.add_subplot(gs6[0, 0])
sc_6a = ax_6a.scatter(
    df_gear_dea["Total_kW"] / 1000.0,
    df_gear_dea["Catch_t"] / 1000.0,
    s=np.sqrt(df_gear_dea["Total_GT"]) * 5.0 + 45,
    c=df_gear_dea["Eco_Efficiency"],
    cmap="viridis", vmin=0.74, vmax=1.00,
    edgecolor="#0F172A", linewidth=1.0, alpha=0.9, zorder=5,
)
coords_a = {r["Clean_Label"]: (r["Total_kW"] / 1000.0, r["Catch_t"] / 1000.0) for _, r in df_gear_dea.iterrows()}

for lbl, txt_xy, ha_v, sh_b in [
    ("East Purse Seine (≥24m)",  (95.0, 165.0), "left",   16),
    ("West Purse Seine (≥24m)",  (66.0, 62.0),  "left",   15),
    ("East Beam Trawl (6–12m)",  (24.3, 44.0),  "center", 10),
    ("East Small-Scale (6–12m)", (98.0, 18.0),  "left",   12),
    ("West Small-Scale (6–12m)", (142.0, 25.0), "center", 12),
    ("East Purse Seine (12–18m)",(4.0, 68.0),   "left",   7),
    ("West Beam Trawl (6–12m)",  (4.0, 54.0),   "left",   7),
]:
    ax_6a.annotate(
        lbl, xy=coords_a[lbl], xytext=txt_xy, fontsize=8.2, weight="bold", ha=ha_v,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.92, edgecolor="#CBD5E1", lw=0.5),
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#334155", shrinkA=2, shrinkB=sh_b),
    )

for name, y_pos, shB in [
    ("East Trawler (12–18m)", 40.0, 8),
    ("West Trawler (18–24m)", 32.0, 11),
    ("East Trawler (≥24m)",   24.0, 13),
    ("East Trawler (18–24m)", 16.0, 12),
    ("West Trawler (12–18m)",  8.0, 11),
]:
    ax_6a.annotate(
        name, xy=coords_a[name], xytext=(55.0, y_pos), fontsize=7.8, weight="bold", ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.92, edgecolor="#CBD5E1", lw=0.5),
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#334155", shrinkA=2, shrinkB=shB),
    )

ax_6a.annotate(
    "Low-Aggregate-Power Cohorts (<7 MW):\n• E/W Purse Seine (12–24m)\n• E/W Beam Trawl (12–18m)\n• E/W Small-Scale (12–24m)\n• W. Trawler (≥24m)",
    xy=(4.0, 2.0), xytext=(4.0, 96.0), fontsize=7.9, ha="left", va="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8FAFC", alpha=0.95, edgecolor="#94A3B8", lw=0.7),
    arrowprops=dict(arrowstyle="->", lw=0.9, color="#475569", shrinkA=2, shrinkB=5),
)

h_6a = [ax_6a.plot([], [], "o", ms=np.sqrt(np.sqrt(gt) * 5.0 + 45), color="#94A3B8", markeredgecolor="#0F172A", markeredgewidth=0.8)[0] for gt in [1000, 10000, 25000]]
ax_6a.legend(handles=h_6a, labels=["1,000 GT", "10,000 GT", "25,000 GT"], loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.8, title="Cohort Tonnage", title_fontsize=9.2)

cax_6a = ax_6a.inset_axes([0.48, 0.72, 0.44, 0.038])
cbar_6a = fig6.colorbar(sc_6a, cax=cax_6a, orientation="horizontal")
cbar_6a.set_label("Eco-Efficiency Score (1 / [1 + β])", fontsize=8.8, weight="bold")
cbar_6a.ax.tick_params(labelsize=7.8)

ax_6a.set_title("(a) 21 Operational Cohorts: Certified Power vs. Guild Landings (n = 4,772)", fontsize=11.4, weight="bold", loc="left", pad=18)
ax_6a.set_xlabel("Total Certified Engine Power (MW)", fontsize=10.5, weight="bold")
ax_6a.set_ylabel("Mean Annual Landings (Thousand Tonnes)", fontsize=10.5, weight="bold")
ax_6a.set_xlim(-5, 168)
ax_6a.set_ylim(-3, 185)
ax_6a.grid(True, linestyle=":", alpha=0.5)

# Panel B: Technical Efficiency (VRS vs CRS) & Eco-Efficiency across 21 Cohorts
ax_6b = fig6.add_subplot(gs6[0, 1])
y_g = np.arange(len(df_gear_dea))
h_bar = 0.27
clean_labels_b_inv = clean_labels_b[::-1]
te_crs_inv = df_gear_dea["TE_CRS"].values[::-1]
te_vrs_inv = df_gear_dea["TE_VRS"].values[::-1]
eco_inv = df_gear_dea["Eco_Efficiency"].values[::-1]

ax_6b.barh(y_g + h_bar, te_vrs_inv, h_bar, label="VRS Technical Efficiency (11/21 Frontier)", color="#0284C7", alpha=0.88, edgecolor="#1E293B", linewidth=0.4)
ax_6b.barh(y_g, te_crs_inv, h_bar, label="CRS Technical Efficiency (10/21 Frontier)", color="#6366F1", alpha=0.85, edgecolor="#1E293B", linewidth=0.4)
ax_6b.barh(y_g - h_bar, eco_inv, h_bar, label="Directional Eco-Efficiency", color="#10B981", alpha=0.88, edgecolor="#1E293B", linewidth=0.4)
ax_6b.axvline(1.0, color="#DC2626", linestyle="--", linewidth=1.2, zorder=4)

ax_6b.set_yticks(y_g)
ax_6b.set_yticklabels(clean_labels_b_inv, fontsize=7.9, weight="bold")
ax_6b.set_title("(b) VRS, CRS and Directional Eco-Efficiency Scores Across 21 Cohorts", fontsize=11.4, weight="bold", loc="left", pad=30)
ax_6b.set_xlabel("Efficiency Score (0.0 to 1.0 Frontier)", fontsize=10.5, weight="bold")
ax_6b.set_xlim(0.0, 1.12)
ax_6b.legend(loc="lower left", bbox_to_anchor=(0.0, 1.015), ncol=3, frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=7.7)
ax_6b.grid(True, linestyle=":", alpha=0.5)

# Panel C: Realistic Duty-Cycle CO2 Emissions vs Directional Abatement Slack (Total = 30.37 kt)
ax_6c = fig6.add_subplot(gs6[1, :])
sc_6c = ax_6c.scatter(
    df_gear_dea["CO2_t"] / 1000.0,
    df_gear_dea["CO2_Abatement_kt"],
    s=np.sqrt(df_gear_dea["Total_GT"]) * 5.5 + 60,
    c=df_gear_dea["Eco_Efficiency"],
    cmap="viridis", vmin=0.74, vmax=1.00,
    edgecolor="#0F172A", linewidth=1.0, alpha=0.9, zorder=5,
)

row_lookup = {r["Clean_Label"]: r for _, r in df_gear_dea.iterrows()}


def annotate_c(lbl, txt_xy, ha_v="center", sh_b=12):
    r = row_lookup[lbl]
    xy = (r["CO2_t"] / 1000.0, r["CO2_Abatement_kt"])
    slk = r["CO2_Abatement_kt"]
    sub = f"[Slack: {slk:.2f} kt]" if slk > 0.01 else "[Frontier: 0.0 kt]"
    ax_6c.annotate(
        f"{lbl}\n{sub}", xy=xy, xytext=txt_xy, fontsize=8.0, weight="bold", ha=ha_v,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.94, edgecolor="#CBD5E1", lw=0.5),
        arrowprops=dict(arrowstyle="->", lw=0.85, color="#0F172A", shrinkA=2, shrinkB=sh_b),
    )


annotate_c("West Purse Seine (≥24m)",  (56.0, 17.2), "left",   16)
annotate_c("East Trawler (18–24m)",    (31.5, 9.2),  "left",   11)
annotate_c("East Trawler (≥24m)",      (23.5, 12.2), "center", 11)
annotate_c("East Trawler (12–18m)",    (10.8, 6.4),  "center", 9)
annotate_c("West Small-Scale (6–12m)", (17.2, 8.8),  "center", 12)
annotate_c("West Purse Seine (12–18m)",(1.8,  9.4),  "left",   7)
annotate_c("West Purse Seine (18–24m)",(1.8,  5.2),  "left",   7)

annotate_c("East Purse Seine (≥24m)",  (76.5, 4.2),  "center", 17)
annotate_c("West Trawler (12–18m)",    (36.0, 2.2),  "left",   10)
annotate_c("West Trawler (18–24m)",    (24.5, 1.9),  "center", 10)
annotate_c("East Small-Scale (6–12m)", (14.2, 4.1),  "center", 11)
annotate_c("East Beam Trawl (6–12m)",  (6.2,  2.2),  "center", 9)

ax_6c.annotate(
    r"Total DEA Sample Duty-Cycle Emissions: 305.59 kt $\mathrm{CO}_2$/yr (21 Cohorts, n = 4,772)" + "\n"
    r"Aggregate Directional Technical Slack: 30.37 kt $\mathrm{CO}_2$/yr (9.94% of Cohort Emissions)",
    xy=(52.0, 14.5), xytext=(44.0, 14.2), fontsize=8.8, weight="bold", color="#0F172A",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#ECFDF5", alpha=0.96, edgecolor="#047857", lw=0.9),
)

cbar_6c = plt.colorbar(sc_6c, ax=ax_6c, orientation="vertical", fraction=0.018, pad=0.02, aspect=20)
cbar_6c.set_label("Eco-Efficiency Score (1 / [1 + β])", fontsize=10.5, weight="bold")

ax_6c.set_title(r"(c) Operational Duty-Cycle $\mathrm{CO}_2$ Emissions vs. Directional Abatement Slack Across 21 Cohorts", fontsize=11.8, weight="bold", loc="left")
ax_6c.set_xlabel(r"Cohort Operational Baseline $\mathrm{CO}_2$ Emissions (Thousand Tonnes $\mathrm{CO}_2$ / Year)", fontsize=10.8, weight="bold")
ax_6c.set_ylabel(r"Directional Abatement Slack (Thousand Tonnes $\mathrm{CO}_2$ / Year)", fontsize=10.8, weight="bold")
ax_6c.set_xlim(-2, 86)
ax_6c.set_ylim(-0.9, 19.8)
ax_6c.grid(True, linestyle=":", alpha=0.6)

save_pub_fig(fig6, "fig7_dea_eco_efficiency_frontiers")


# ==============================================================================
# FIGURE 6: DECARBONIZATION POLICY SCENARIOS, 2023 SHOCKS & PORTFOLIO RISK
# ==============================================================================
print("\n--> Generating Figure 6: Decarbonization Policy Scenarios, 2023 Shocks & Portfolio Risk...")

fig7, axes7 = plt.subplots(2, 2, figsize=(14.6, 10.4), dpi=600)

# Panel A: Realistic Operational Duty-Cycle CO2 Abatement Across Scenarios
ax_7a = axes7[0, 0]
scen_labels = [
    "Baseline\n(Duty Cycle)",
    "Scen. A\n441 kW Limit",
    "Scen. B\n>35 yr Exit",
    "Scen. C\nGear Shift",
    "Scen. D\n-15% Tech",
    "Scen. E\nIntegrated",
    "Sens. E750\n750 kW PS",
]
co2_vals = df_scen["Realistic_CO2_Emissions_kt"].values
co2_red = df_scen["Realistic_CO2_Reduction_%"].values
bar_cols_7a = ["#475569", "#0284C7", "#10B981", "#D97706", "#8B5CF6", "#DC2626", "#E11D48"]

bars_7a = ax_7a.bar(range(len(scen_labels)), co2_vals, color=bar_cols_7a, edgecolor="#1E293B", linewidth=0.6, alpha=0.88, width=0.62)
for idx, bar in enumerate(bars_7a):
    h = bar.get_height()
    red_txt = f"-{co2_red[idx]:.1f}%" if co2_red[idx] > 0 else "Base"
    ax_7a.text(bar.get_x() + bar.get_width() / 2, h + 7, f"{h:.1f} kt\n({red_txt})", ha="center", fontsize=8.4, weight="bold")

ax_7a.annotate(
    "Uniform Capacity Upper-Bound Envelope (1.1616 t/kW/yr):\n784.9 kt Base → 618.0 kt (-21.3%, -167.0 kt) in Scenario E",
    xy=(5, 259.4), xytext=(1.15, 412.0),
    fontsize=8.3, weight="bold", color="#1E293B",
    bbox=dict(boxstyle="round,pad=0.28", facecolor="#F8FAFC", edgecolor="#94A3B8", lw=0.7),
)
ax_7a.set_xticks(range(len(scen_labels)))
ax_7a.set_xticklabels(scen_labels, fontsize=8.6, weight="bold")
ax_7a.set_title("(a) Conditional Policy Scenarios: Operational Duty-Cycle CO$_2$ Inventory", fontsize=11.6, weight="bold", loc="left")
ax_7a.set_ylabel("Annual Fleet CO$_2$ Emissions (kt / yr)", fontsize=10.8, weight="bold")
ax_7a.set_ylim(0, 475)

# Panel B: Exact 2023 TÜİK Fuel Shock Sensitivity + 8%-20% Depreciation Band
ax_7b = axes7[0, 1]
df_sens_plot = df_sens.copy()
df_sens_plot["Shock_Pct"] = df_sens_plot["Fuel_Price_Shock"].apply(
    lambda s: 0.0 if "Baseline" in str(s) else float(str(s).replace("+", "").replace("%", "").strip())
)
df_sens_plot = df_sens_plot[df_sens_plot["Shock_Pct"] <= 100].copy()
shocks = df_sens_plot["Shock_Pct"].values
gross_m = df_sens_plot["Gross_Cash_Operating_Margin_%"].values
net_m_15 = df_sens_plot["Depreciated_Net_Margin_15pct_%"].values
net_m_08 = gross_m - 8.0
net_m_20 = gross_m - 20.0

ax_7b.plot(shocks, gross_m, color="#0284C7", marker="s", linewidth=2.2, label="2023 Gross Cash Margin (Base: 44.52%)")
ax_7b.plot(shocks, net_m_15, color="#DC2626", marker="o", linewidth=2.2, linestyle="--", label=r"2023 Depreciated Net Margin ($\delta = 15\%$: 29.52%)")
ax_7b.fill_between(shocks, net_m_20, net_m_08, color="#FEE2E2", alpha=0.55, label=r"Depreciation Sensitivity Band ($\delta = 8\%$ to $20\%$)")
ax_7b.axhline(0.0, color="#1E293B", linestyle="-", linewidth=1.0)
ax_7b.annotate(
    "Zero-Net-Margin Break-Even:\n+142.6% Diesel Surge (at $\\delta = 15\\%$)",
    xy=(100, 8.81), xytext=(45, 4.0),
    arrowprops=dict(facecolor="#B91C1C", arrowstyle="->", lw=1.1),
    fontsize=8.5, weight="bold", color="#B91C1C",
    bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#B91C1C", lw=0.6),
)

ax_7b.set_title("(b) 2023 TÜİK Operating Margins Under Diesel Price Shocks", fontsize=11.6, weight="bold", loc="left")
ax_7b.set_xlabel("Simulated Real Marine Diesel Price Increase (%)", fontsize=10.8, weight="bold")
ax_7b.set_ylabel("Operating Margin (% of Gross Revenue)", fontsize=10.8, weight="bold")
ax_7b.set_ylim(-5, 52)
ax_7b.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.4)

# Panel C: Surviving-Fleet Monte Carlo Net Margins (0% Rebound vs +5% Crowding Relief)
ax_7c = axes7[1, 0]
sns.kdeplot(df_mc["Scenario_B_ZeroRebound_Margin"], ax=ax_7c, color="#10B981", linestyle="--", linewidth=1.7,
            label=f"Scen. B0: >35 yr Exit, 0% Rebound ({df_mc['Scenario_B_ZeroRebound_Margin'].mean():.2f}%)")
sns.kdeplot(df_mc["Scenario_B_Margin"], ax=ax_7c, color="#047857", linestyle="-", linewidth=1.9,
            label=f"Scen. B: >35 yr Exit, +5% CPUE Relief ({df_mc['Scenario_B_Margin'].mean():.2f}%)")
sns.kdeplot(df_mc["Scenario_A_Margin"], ax=ax_7c, color="#0284C7", linestyle="-", linewidth=1.8,
            label=f"Scen. A: 441 kW Limit ({df_mc['Scenario_A_Margin'].mean():.2f}%)")
sns.kdeplot(df_mc["Scenario_E_ZeroRebound_Margin"], ax=ax_7c, color="#F97316", linestyle="--", linewidth=1.8,
            label=f"Scen. E0: Integrated, 0% Rebound ({df_mc['Scenario_E_ZeroRebound_Margin'].mean():.2f}%)")
sns.kdeplot(df_mc["Scenario_E_Combined_Margin"], ax=ax_7c, color="#DC2626", linestyle="-", linewidth=2.2,
            label=f"Scen. E: Integrated, +5% Relief ({df_mc['Scenario_E_Combined_Margin'].mean():.2f}%)")
ax_7c.axvline(29.52, color="#1E293B", linestyle=":", linewidth=1.5, label="2023 Baseline Net Margin (29.52%)")

ax_7c.set_title("(c) Surviving-Fleet Net Margins: 0% vs. +5% CPUE Rebound ($N = 10,000$)", fontsize=11.6, weight="bold", loc="left")
ax_7c.set_xlabel("Surviving-Fleet Depreciated Net Operating Margin (%)", fontsize=10.8, weight="bold")
ax_7c.set_ylabel("Monte Carlo Kernel Density", fontsize=10.8, weight="bold")
ax_7c.set_xlim(25.5, 39.5)
ax_7c.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.0)

# Panel D: Exact Model 3 Marginal Effect of Shannon Diversity Across Fleet Scales
ax_7d = axes7[1, 1]
reg_df = panel_df.dropna(subset=["Rev_Roll_CV", "Shannon_Roll_Mean", "Vessels_Roll_Mean"]).copy()
reg_df["ln_V"] = np.log(reg_df["Vessels_Roll_Mean"])
mean_lnv = reg_df["ln_V"].mean()  # 7.9827
p10_lnv = reg_df["ln_V"].quantile(0.10)
p90_lnv = reg_df["ln_V"].quantile(0.90)

# Scatter actual 105 rolling observations colored by fleet scale
sc_7d = ax_7d.scatter(
    reg_df["Shannon_Roll_Mean"], reg_df["Rev_Roll_CV"],
    c=reg_df["ln_V"], cmap="coolwarm", s=34, alpha=0.68, edgecolor="#1E293B", linewidth=0.4,
)

# Exact Model 3 coefficients: dCV/dH' = -2.1695 + 0.2648 * ln_V
h_grid = np.linspace(reg_df["Shannon_Roll_Mean"].min(), reg_df["Shannon_Roll_Mean"].max(), 100)
h_mean = reg_df["Shannon_Roll_Mean"].mean()
cv_mean = reg_df["Rev_Roll_CV"].mean()

for lnv_val, col_l, ls_l, tag in [
    (p10_lnv,  "#15803D", "-",  f"Low Fleet Scale (P10 ln V = {p10_lnv:.2f}, Slope = {-2.1695 + 0.2648*p10_lnv:+.3f})"),
    (mean_lnv, "#0284C7", "--", f"Mean Fleet Scale (Mean ln V = {mean_lnv:.2f}, Slope = {-2.1695 + 0.2648*mean_lnv:+.3f})"),
    (p90_lnv,  "#DC2626", ":",  f"High Fleet Scale (P90 ln V = {p90_lnv:.2f}, Slope = {-2.1695 + 0.2648*p90_lnv:+.3f})"),
]:
    slope_v = -2.1695 + 0.2648 * lnv_val
    level_shift = 0.0654 * (lnv_val - mean_lnv) + 0.2648 * h_mean * (lnv_val - mean_lnv)
    y_pred = cv_mean + level_shift + slope_v * (h_grid - h_mean)
    ax_7d.plot(h_grid, y_pred, color=col_l, linestyle=ls_l, linewidth=2.2, label=tag)

ax_7d.set_title("(d) Catch Species Diversity vs. Revenue Volatility Across Fleet Scales (Model 3)", fontsize=11.4, weight="bold", loc="left")
ax_7d.set_xlabel("Rolling 5-Year Shannon Catch Diversity Index ($H'$)", fontsize=10.8, weight="bold")
ax_7d.set_ylabel("Rolling 5-Year Revenue Coefficient of Variation", fontsize=10.8, weight="bold")
ax_7d.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.0)

plt.tight_layout(pad=1.4)
save_pub_fig(fig7, "fig6_decarbonisation_tradeoffs")


# ==============================================================================
# FIGURE S1: SUPPLEMENTARY AGE STRUCTURE, REAL MORAN SCATTER & CO2 SENSITIVITY
# ==============================================================================
print("\n--> Generating Supplementary Figure S1...")

figS1, axesS1 = plt.subplots(2, 2, figsize=(14.4, 10.2), dpi=600)

# Panel A: Age Structure
ax_s1a = axesS1[0, 0]
sns.histplot(tr_fleet["vessel_age"].dropna(), bins=35, ax=ax_s1a, color="#0284C7", edgecolor="#1E293B", alpha=0.8, kde=True)
ax_s1a.axvline(35.0, color="#DC2626", linestyle="--", linewidth=1.5, label="Retirement Threshold (>35 Years: 9.99%, n = 658)")
ax_s1a.axvline(tr_fleet["vessel_age"].mean(), color="#1E293B", linestyle=":", linewidth=1.5, label=f"Fleet Mean Age ({tr_fleet['vessel_age'].mean():.1f} Years)")
ax_s1a.set_title("(a) Age Distribution of Registered Turkish GSA 29 Vessels (N = 6,584)", fontsize=11.6, weight="bold", loc="left")
ax_s1a.set_xlabel("Vessel Service Age (Years)", fontsize=10.8, weight="bold")
ax_s1a.set_ylabel("Registered Vessel Count", fontsize=10.8, weight="bold")
ax_s1a.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9.2)

# Panel B: Vessel Age vs EII (Full N = 6,584)
ax_s1b = axesS1[0, 1]
sub_age = tr_fleet[(tr_fleet["eii"] < 65) & (tr_fleet["vessel_age"].notna())].copy()
sns.regplot(data=sub_age, x="vessel_age", y="eii", ax=ax_s1b,
            scatter_kws={"alpha": 0.22, "s": 13, "color": "#64748B"},
            line_kws={"color": "#DC2626", "linewidth": 2.0, "label": "OLS Linear Fit"})
ax_s1b.axhline(6.0, color="#1E293B", linestyle="--", linewidth=1.2, label=r"Purse Seine Upper Benchmark (6.0 kW GT$^{-1}$)")
ax_s1b.axhline(10.0, color="#D97706", linestyle=":", linewidth=1.2, label=r"Trawler Upper Benchmark (10.0 kW GT$^{-1}$)")
ax_s1b.set_title("(b) Vessel Age vs. Engine Intensity Across Registered Vessels", fontsize=11.6, weight="bold", loc="left")
ax_s1b.set_xlabel("Vessel Service Age (Years)", fontsize=10.8, weight="bold")
ax_s1b.set_ylabel(r"Engine Intensity Index ($\mathrm{kW\ GT}^{-1}$)", fontsize=10.8, weight="bold")
ax_s1b.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9.0)

# Panel C: Empirical Moran Scatter Plot on 27 Black Sea Coastal Ports
ax_s1c = axesS1[1, 0]
z_real = df_moran_pts["z_Median_EII"].values
wz_real = df_moran_pts["Wz_Spatial_Lag_EII"].values

quad_colors = {
    "HH (High-Intensity Trawl Cluster)": "#DC2626",
    "LL (Low-Intensity Pelagic Zone)": "#0284C7",
    "LH (Low-Port in High Zone)": "#D97706",
    "HL (High-Port in Low Zone)": "#7E22CE",
    "Not Significant": "#94A3B8",
}

for _, rp in df_moran_pts.iterrows():
    sig = rp["p_sim"] < 0.05
    zv = rp["z_Median_EII"]
    wzv = rp["Wz_Spatial_Lag_EII"]
    if sig and zv >= 0 and wzv >= 0:
        grp = "HH (High-Intensity Trawl Cluster)"
    elif sig and zv < 0 and wzv < 0:
        grp = "LL (Low-Intensity Pelagic Zone)"
    elif sig and zv < 0 and wzv >= 0:
        grp = "LH (Low-Port in High Zone)"
    elif sig and zv >= 0 and wzv < 0:
        grp = "HL (High-Port in Low Zone)"
    else:
        grp = "Not Significant"
    ax_s1c.scatter(zv, wzv, color=quad_colors[grp], edgecolor="#0F172A", s=58, alpha=0.90, zorder=5)

# Regression line slope = exact Moran's I = +0.6408
z_line = np.linspace(z_real.min() - 0.2, z_real.max() + 0.2, 100)
ax_s1c.plot(z_line, 0.6408 * z_line, color="#B91C1C", linewidth=2.1, label="Exact Moran's I = +0.641 (z = 5.65, p = 0.0001)")
ax_s1c.axhline(0, color="#94A3B8", linestyle="--", linewidth=0.9)
ax_s1c.axvline(0, color="#94A3B8", linestyle="--", linewidth=0.9)

moran_lbl_offsets = {
    "KARASU":      ( 0.06,  0.06, "left"),
    "KDZ. EREGLI": (-0.06, -0.15, "right"),
    "KEFKEN":      ( 0.06,  0.09, "left"),
    "BARTIN":      (-0.06,  0.09, "right"),
    "CIDE":        ( 0.06, -0.04, "left"),
    "HOPA":        (-0.07,  0.07, "right"),
    "PAZAR":       ( 0.07,  0.02, "left"),
    "RIZE":        ( 0.06, -0.14, "left"),
    "TRABZON":     ( 0.06,  0.06, "left"),
    "ISTANBUL":    ( 0.06,  0.05, "left"),
    "ZONGULDAK":   ( 0.06, -0.04, "left"),
    "SAMSUN":      ( 0.06,  0.06, "left"),
}
for p_lbl, (dx_m, dy_m, ha_m) in moran_lbl_offsets.items():
    sub_p = df_moran_pts[df_moran_pts["Port"] == p_lbl]
    if len(sub_p) > 0:
        rp = sub_p.iloc[0]
        clean_p = PORT_CLEAN_NAMES.get(p_lbl, p_lbl.title())
        ax_s1c.text(rp["z_Median_EII"] + dx_m, rp["Wz_Spatial_Lag_EII"] + dy_m, clean_p, ha=ha_m, fontsize=7.6, weight="bold", color="#1E293B")

legend_handles_c = [
    plt.Line2D([0], [0], color="#B91C1C", lw=2.1, label="Moran's I = +0.641 (p = 0.0001)"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#DC2626", markeredgecolor="#0F172A", markersize=7, label="Significant HH (West Shelf, p < 0.05)"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#0284C7", markeredgecolor="#0F172A", markersize=7, label="Significant LL (East Pelagic, p < 0.05)"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#D97706", markeredgecolor="#0F172A", markersize=7, label="Significant LH Outlier (p < 0.05)"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#94A3B8", markeredgecolor="#0F172A", markersize=7, label="Non-Significant Port (p ≥ 0.05)"),
]
ax_s1c.set_xlim(-1.85, 2.38)
ax_s1c.set_ylim(-1.42, 1.72)
ax_s1c.set_title("(c) Empirical Moran Scatter Plot: 27 Black Sea Ports ($k = 4$ Nearest)", fontsize=11.6, weight="bold", loc="left")
ax_s1c.set_xlabel("Standardized Port Median Engine Intensity ($z_i$)", fontsize=10.8, weight="bold")
ax_s1c.set_ylabel(r"Row-Standardized Spatial Lag ($\sum_j w_{ij} z_j$)", fontsize=10.8, weight="bold")
ax_s1c.legend(handles=legend_handles_c, loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.0)

# Panel D: Parametric Sensitivity Comparing Primary Realistic Baseline (367.0 kt) vs Uniform Bound (784.9 kt)
ax_s1d = axesS1[1, 1]
hours_range = np.linspace(600, 2000, 60)
for eff_lf, color, label in [
    (0.35, "#15803D", r"Low Effective Load ($LF \times \kappa_{\mathrm{aux}} = 0.35$)"),
    (0.65, "#0284C7", r"Moderate Effective Load ($LF \times \kappa_{\mathrm{aux}} = 0.65$)"),
    (1.09, "#DC2626", r"Heavy Industrial + Aux Load ($LF \times \kappa_{\mathrm{aux}} = 1.09$)"),
]:
    co2_curve = (675_722.62 * eff_lf * 0.210 * hours_range * 3.17) / 1e6
    ax_s1d.plot(hours_range, co2_curve, color=color, linewidth=2.0, label=label)

ax_s1d.axhline(366.98, color="#047857", linestyle="-.", linewidth=2.0, label="Primary Gear-Differentiated Duty-Cycle Baseline (367.0 kt)")
ax_s1d.axhline(784.92, color="#1E293B", linestyle="--", linewidth=1.4, label="Uniform Capacity Sensitivity Upper Bound (784.9 kt at 1,600 h)")

ax_s1d.set_title("(d) Parametric CO$_2$ Sensitivity: Duty-Cycle Baseline vs. Uniform Bounds", fontsize=11.6, weight="bold", loc="left")
ax_s1d.set_xlabel("Hypothetical Fleet-Wide Uniform Operating Hours (h / yr)", fontsize=10.8, weight="bold")
ax_s1d.set_ylabel("Annual Fleet CO$_2$ Emissions (kt / yr)", fontsize=10.8, weight="bold")
ax_s1d.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.2)

plt.tight_layout(pad=1.4)
save_pub_fig(figS1, "figS1_supplementary_diagnostics")

print("\n[SUCCESS] Publication figures (Figures 2-7 and Figure S1) generated at 600 DPI.")


# ==============================================================================
# FIGURE 1: INTEGRATED METHODOLOGICAL WORKFLOW DIAGRAM
# ==============================================================================
import textwrap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def generate_figure_1_workflow():
    fig = plt.figure(figsize=(12.2, 8.9), dpi=600)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    c_phase1 = "#1E3A8A"
    c_phase2 = "#0369A1"
    c_phase3 = "#D97706"
    c_phase4 = "#047857"
    c_phase5 = "#B91C1C"
    c_phase6 = "#6D28D9"
    c_phase7 = "#0F766E"

    bg_phase1 = "#F8FAFC"
    bg_phase2 = "#F0F9FF"
    bg_phase3 = "#FFFBEB"
    bg_phase4 = "#ECFDF5"
    bg_phase5 = "#FEF2F2"
    bg_phase6 = "#F5F3FF"
    bg_phase7 = "#F0FDFA"

    ax.text(
        50, 98.0,
        "INTEGRATED EMPIRICAL, ENGINEERING AND POLICY WORKFLOW FRAMEWORK",
        ha="center", va="center", fontsize=12.4, fontweight="bold",
        color="#0F172A", fontfamily="DejaVu Sans",
    )
    ax.text(
        50, 96.0,
        "Observed Registry Microdata, Semi-Empirical Naval Resistance, Spatial Diagnostics, Frontier Efficiency and Conditional Policy Scenarios",
        ha="center", va="center", fontsize=8.6, fontstyle="italic",
        color="#475569", fontfamily="DejaVu Sans",
    )

    def draw_card(
        x, y, w, h, title, items, header_color, bg_color, border_color,
        title_size=8.8, body_size=7.15, line_step=1.28, item_gap=0.16,
        header_h=2.8, top_pad=1.0,
    ):
        card = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
            facecolor=bg_color, edgecolor=border_color, linewidth=1.1, zorder=2,
        )
        ax.add_patch(card)

        header_box = FancyBboxPatch(
            (x, y + h - header_h), w, header_h,
            boxstyle="round,pad=0.15,rounding_size=0.5",
            facecolor=header_color, edgecolor=header_color, linewidth=0.5, zorder=3,
        )
        ax.add_patch(header_box)
        ax.text(
            x + w / 2.0, y + h - (header_h / 2.0), title,
            ha="center", va="center", fontsize=title_size,
            fontweight="bold", color="#FFFFFF", zorder=4,
        )

        y_pos = y + h - header_h - top_pad
        max_chars = int((w - 2.0) / 0.435)

        for item in items:
            prefix = ""
            indent = 0.8
            style_bold = False
            text_color = "#1E293B"

            if item.startswith("• "):
                prefix = "• "
                raw_text = item[2:]
            elif item.startswith("  – "):
                prefix = "  – "
                raw_text = item[4:]
                indent = 2.0
                text_color = "#334155"
            else:
                raw_text = item

            wrapped_lines = textwrap.wrap(raw_text, width=max_chars - len(prefix))
            if not wrapped_lines:
                y_pos -= line_step
                continue

            for idx, wline in enumerate(wrapped_lines):
                line_str = (prefix if idx == 0 else "   ") + wline
                ax.text(
                    x + indent, y_pos, line_str, ha="left", va="top",
                    fontsize=body_size,
                    fontweight="bold" if style_bold else "normal",
                    color=text_color, zorder=4,
                )
                y_pos -= line_step
            y_pos -= item_gap

    def draw_arrow(x1, y1, x2, y2, color, label="", rad=0.0, label_offset=(0, 0), font_size=6.8):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=10, linewidth=1.2,
            color=color, connectionstyle=f"arc3,rad={rad}", zorder=5,
        )
        ax.add_patch(arrow)
        if label:
            xm = (x1 + x2) / 2.0 + label_offset[0]
            ym = (y1 + y2) / 2.0 + label_offset[1]
            ax.text(
                xm, ym, label, ha="center", va="center",
                fontsize=font_size, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.20", fc="#FFFFFF", ec=color, lw=0.6, alpha=0.96),
                zorder=6,
            )

    # TIER 1: OBSERVED ADMINISTRATIVE & STATISTICAL INPUTS
    card_w1 = 22.2
    y1 = 81.8
    h1 = 12.2

    draw_card(
        2.5, y1, card_w1, h1,
        "1A. Observed Fleet Register",
        [
            "• Subarea 29 vessel register microdata",
            "• 6,584 active Turkish vessels across 34 ports",
            "• 675.7 megawatts and 87,364 gross tons",
            "• Purse seiners, trawlers, dredgers, small boats",
            "• Mean age 22.6 years, 9.99 percent over 35",
        ],
        c_phase1, bg_phase1, c_phase1,
        title_size=8.5, body_size=7.10, line_step=1.35, item_gap=0.18, header_h=2.6, top_pad=0.9,
    )

    draw_card(
        27.0, y1, card_w1, h1,
        "1B. Observed Catch Accounts",
        [
            "• National Black Sea catch 256.7 thousand tons",
            "• Area 37 regional context 1950 to 2024",
            "• Area 37 cumulative 41.9 million tons",
            "• Pelagic, demersal and mollusc species guilds",
            "• Anchovy, sprat, bonito and sea snail",
        ],
        c_phase1, bg_phase1, c_phase1,
        title_size=8.5, body_size=7.10, line_step=1.35, item_gap=0.18, header_h=2.6, top_pad=0.9,
    )

    draw_card(
        51.5, y1, card_w1, h1,
        "1C. Observed Cost Accounts",
        [
            "• National marine fishery cost structure 2023",
            "• Fuel represents 20.70 percent of revenue",
            "• Crew share labor equals 19.00 percent",
            "• Gross cash operating margin is 44.52 percent",
            "• Depreciated net margin is 29.52 percent",
        ],
        c_phase1, bg_phase1, c_phase1,
        title_size=8.5, body_size=7.10, line_step=1.35, item_gap=0.18, header_h=2.6, top_pad=0.9,
    )

    draw_card(
        75.8, y1, card_w1, h1,
        "1D. Macro and Ocean Series",
        [
            "• Central Bank annual exchange rate series",
            "• Consumer price index inflation deflators",
            "• Basin sea surface temperature anomalies",
            "• Satellite surface chlorophyll series",
            "• Illustrative satellite speed check subset",
        ],
        c_phase1, bg_phase1, c_phase1,
        title_size=8.5, body_size=7.10, line_step=1.35, item_gap=0.18, header_h=2.6, top_pad=0.9,
    )

    # TIER 2: DIAGNOSTIC & SEMI-EMPIRICAL MODELS
    y2 = 60.0
    h2 = 16.5
    w2 = 30.0

    draw_card(
        2.5, y2, w2, h2,
        "PHASE 2: Semi-Empirical Naval Hydrodynamics",
        [
            "• Fleet mean engine intensity is 14.53 kilowatts per ton",
            "• Gear-specific displacement and bollard benchmarks",
            "• 43.9 percent of trawlers exceed 10 kilowatts per ton",
            "• Holtrop-Mennen and frictional resistance decomposition",
            "• Transit wave drag reaches 58 to 79 percent of resistance",
            "• Active towing wave drag remains under 1.3 percent",
            "• Displacement-normalized power checks confirm ranking",
            "• Within-gear scaling slopes remain negative for large vessels",
        ],
        c_phase2, bg_phase2, c_phase2,
        title_size=8.6, body_size=7.05, line_step=1.30, item_gap=0.15, header_h=2.6, top_pad=0.9,
    )

    draw_card(
        35.0, y2, w2, h2,
        "PHASE 3: Spatial Port Autocorrelation Analysis",
        [
            "• Full 34-port audit: 27 Black Sea plus 7 Marmara ports",
            "• Four-nearest-neighbor coastal weight matrix",
            "• Port engine intensity Moran index is 0.641",
            "• Dual local spatial cluster testing with false discovery:",
            "  – High-intensity western shelf trawling cluster",
            "  – Low-intensity eastern pelagic purse-seine zone",
            "• Aggregate port power shows dispersed hub structure",
            "• Priority industrial hubs identified for shore power",
        ],
        c_phase3, bg_phase3, c_phase3,
        title_size=8.6, body_size=7.05, line_step=1.30, item_gap=0.15, header_h=2.6, top_pad=0.9,
    )

    draw_card(
        67.5, y2, w2, h2,
        "PHASE 4: Cohort Frontier Eco-Efficiency Model",
        [
            "• Four-step exclusion audit yields 4,772 Black Sea vessels",
            "• 21 basin, gear and length operational cohorts",
            "• Inputs: certified engine power and gross tonnage",
            "• Desirable outputs: pelagic, demersal and mollusc catch",
            "• Undesirable output: 305.6 thousand tons carbon dioxide",
            "• Directional distance frontier with constant-returns check",
            "• Single-output dimensionality and allocation robustness",
            "• Technical slack equals 30.4 thousand tons per year",
        ],
        c_phase4, bg_phase4, c_phase4,
        title_size=8.6, body_size=7.05, line_step=1.30, item_gap=0.15, header_h=2.6, top_pad=0.9,
    )

    # TIER 3: CONDITIONAL SCENARIOS & ECONOMETRIC RISK
    y3 = 31.4
    h3 = 22.8
    w3 = 46.0

    draw_card(
        2.5, y3, w3, h3,
        "PHASE 5: Duty-Cycle Carbon Inventory and Policy Scenarios",
        [
            "• Gear-differentiated operational baseline: 367.0 thousand tons",
            "• Uniform industrial capacity upper bound: 784.9 thousand tons",
            "  – Scenario A: 441-kilowatt limit across 315 industrial vessels",
            "  – Scenario B: Retirement of 658 vessels over 35 years old",
            "  – Scenario C: Shift of 165 trawl vessels to passive gear",
            "  – Scenario D: 15-percent technical propulsion efficiency gain",
            "  – Scenario E: Integrated limit and retirement with 42-vessel overlap",
            "• Surviving-fleet Monte Carlo simulation with 10,000 draws:",
            "  – Contrasts zero catch rebound against five-percent relief",
            "• Fuel shock stress tests across 8 to 20 percent depreciation",
            "• Integrated abatement: 107.6 thousand tons or 29.3 percent",
        ],
        c_phase5, bg_phase5, c_phase5,
        title_size=8.8, body_size=7.20, line_step=1.38, item_gap=0.18, header_h=2.8, top_pad=0.9,
    )

    draw_card(
        51.5, y3, w3, h3,
        "PHASE 6: Multi-Basin Panel Portfolio Volatility Analysis",
        [
            "• Five Turkish marine basins observed from 2000 to 2024",
            "• 105 five-year rolling windows plus 25 non-overlapping blocks",
            "• Dependent variables: revenue and catch variation coefficients",
            "• Basin fixed effects with autocorrelation-robust standard errors",
            "• Higher catch species diversity lowers volatility at modest scale",
            "• Positive diversity-by-fleet interaction shows scale erosion",
            "• Mean-centered diversity slope evaluated at sample mean fleet",
            "• Exchange rate depreciation amplifies revenue instability",
            "• Consumer price inflation and thermal anomaly controls included",
            "• Non-overlapping block estimation confirms negative diversity slope",
            "• Fleet rightsizing restores ecological revenue stabilization",
        ],
        c_phase6, bg_phase6, c_phase6,
        title_size=8.8, body_size=7.20, line_step=1.38, item_gap=0.18, header_h=2.8, top_pad=0.9,
    )

    # TIER 4: POLICY GOVERNANCE SYNTHESIS
    y4 = 4.8
    h4 = 19.5
    w4 = 95.0

    draw_card(
        2.5, y4, w4, h4,
        "PHASE 7: Conditional Decarbonization Policy Roadmap and Fisheries Governance",
        [
            "• 1. Certified Engine Power Standards: Phase in a 441-kilowatt main engine limit for trawlers and transitional limits for large seiners",
            "  – Pair continuous shaft-power monitoring with high-thrust nozzles and propeller repitching to maintain low-speed bollard pull",
            "• 2. Targeted Older-Vessel Decommissioning: Retire 658 vessels exceeding 35 years of service alongside strict license-transfer bans",
            "  – Integrated power limits and retirements cut realistic annual emissions by 107.6 thousand tons while raising surviving margins",
            "• 3. Selective Shore-Side Electrification: Deploy cold-ironing connections at high-power industrial ports led by Samsun, Istanbul and Trabzon",
            "  – Target auxiliary generator loads of large purse seiners and refrigerated trawlers during peak seasonal landing queues",
            "• 4. Reformed Fuel Tax Incentives: Redirect unconditional diesel excise exemptions toward verified propulsion and hull upgrades",
            "  – Link modernization co-financing to audited specific fuel oil consumption reductions and low-drag gear adoption",
            "• 5. Multilateral Black Sea Coordination: Harmonize Subarea 29 capacity rules, catch documentation and monitoring across coastal states",
            "  – Shared regional effort discipline prevents cross-border fishing displacement on migratory anchovy and sprat stocks",
        ],
        c_phase7, bg_phase7, c_phase7,
        title_size=9.2, body_size=7.15, line_step=1.25, item_gap=0.22, header_h=2.8, top_pad=0.9,
    )

    # ARROWS
    draw_arrow(13.6, y1, 13.6, y2 + h2, c_phase2, label="Vessel Dimensions and Power")
    draw_arrow(38.0, y1, 42.0, y2 + h2, c_phase3, label="Port Registry and Catch", rad=-0.04)
    draw_arrow(62.5, y1, 72.0, y2 + h2, c_phase4, label="Cost Shares and Guilds", rad=-0.04)
    draw_arrow(86.5, y1, 86.5, y2 + h2, c_phase1, label="Macro and Ocean Series")

    draw_arrow(17.5, y2, 17.5, y3 + h3, c_phase5, label="Transit and Towing Regimes")
    draw_arrow(37.0, y2, 37.0, y3 + h3, c_phase3, label="Port Intensity Clusters")
    draw_arrow(68.5, y2, 47.0, y3 + h3, c_phase5, label="Frontier Abatement Slack", rad=0.06, label_offset=(0, -0.9))
    draw_arrow(80.0, y2, 80.0, y3 + h3, c_phase6, label="Basin Catch and Fleet Series")

    draw_arrow(22.0, y3, 22.0, y4 + h4, c_phase5, label="Conditional Scenarios A to E")
    draw_arrow(50.0, y3, 50.0, y4 + h4, c_phase3, label="Industrial Hub Shore Power")
    draw_arrow(78.0, y3, 78.0, y4 + h4, c_phase6, label="Scale and Diversity Evidence")

    plt.tight_layout()
    save_pub_fig(fig, "fig1_methodological_workflow")
    print("[SUCCESS] Rendered Figure 1 (600 DPI) -> figures/fig1_methodological_workflow.png & .pdf.")


print("\n--> Generating Figure 1: Integrated Methodological Workflow Diagram...")
generate_figure_1_workflow()
