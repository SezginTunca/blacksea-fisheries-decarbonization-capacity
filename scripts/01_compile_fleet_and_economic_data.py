#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT: 01_compile_fleet_and_economic_data.py
STUDY:  Decarbonizing Turkish Black Sea Commercial Fisheries Requires Pairing
        Industrial Engine-Power Limits with Capacity Reduction
JOURNAL: Turkish Journal of Maritime and Marine Sciences (DOI: 10.52998/trjmms.2042258)
AUTHOR: Sezgin Tunca, Ph.D. (ORCID: 0000-0001-6840-1466)
PURPOSE: Compile GSA 29 commercial fleet segment architecture, Black Sea riparian
         regional capacity benchmarks, and longitudinal operational cost accounts
         (2000-2024) for the Turkish Black Sea fishing fleet.
INPUTS:
  - data/gfcm_blacksea_fleet.csv
  - data/turkish_fleet_profitability.csv
OUTPUTS:
  - tables/turkish_fleet_segment_architecture.csv
  - tables/blacksea_regional_fleet_comparison.csv
  - tables/turkish_operational_cost_summary.csv
================================================================================
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TABLES_DIR = os.path.join(BASE_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

# Standard marine fuel carbon emission factor (IMO MEPC / ISO 8178: 3.17 kg CO2/kg fuel)
CO2_PER_KG_FUEL = 3.17
UNIFORM_UPPER_EF = 1.1616  # Uniform industrial upper-bound factor (t CO2 / kW / yr)

# Gear-differentiated operational duty-cycle factors (t CO2 / kW / yr), synchronized with Scripts 03-06
GEAR_EF_REALISTIC = {
    "Purse Seiner (PS)": 1500.0 * 0.55 * 1.75 * 210.0 * CO2_PER_KG_FUEL * 1e-6,
    "Pelagic/Demersal Trawler (OTB/PTM)": 1400.0 * 0.60 * 1.45 * 210.0 * CO2_PER_KG_FUEL * 1e-6,
    "Beam Trawler (TBB)": 900.0 * 0.50 * 1.15 * 215.0 * CO2_PER_KG_FUEL * 1e-6,
    "Dredger (DRB)": 1100.0 * 0.55 * 1.65 * 210.0 * CO2_PER_KG_FUEL * 1e-6,
    "Small-Scale / Polyvalent (MIS)": 600.0 * 0.30 * 1.05 * 220.0 * CO2_PER_KG_FUEL * 1e-6,
}


def classify_turkish_gear(row):
    """Map GFCM primary gear code and gear group into the 5 standard GSA 29 fleet segments."""
    g1 = str(row.get("gear1", "")).strip().upper()
    gg = str(row.get("gear_group", "")).strip().lower()
    if g1 == "PS" or "purse" in gg:
        return "Purse Seiner (PS)"
    elif g1 in ["OTB", "PTM"] or "trawl" in gg:
        if g1 == "TBB":
            return "Beam Trawler (TBB)"
        return "Pelagic/Demersal Trawler (OTB/PTM)"
    elif g1 == "TBB":
        return "Beam Trawler (TBB)"
    elif g1 == "DRB" or "dredg" in gg:
        return "Dredger (DRB)"
    else:
        return "Small-Scale / Polyvalent (MIS)"


def main():
    print("=" * 80)
    print("STEP 01: Compiling GSA 29 Fleet Architecture & Operational Cost Accounts")
    print("=" * 80)

    # 1. Load & filter active commercial vessels with verified physical metrics
    fleet_path = os.path.join(DATA_DIR, "gfcm_blacksea_fleet.csv")
    fleet_df = pd.read_csv(fleet_path, low_memory=False)
    print(f"Total raw regional fleet records loaded: {len(fleet_df):,}")

    fleet_clean = fleet_df[
        np.isfinite(fleet_df["eii"])
        & (fleet_df["tonnage_gt"] > 0)
        & (fleet_df["engine_power_kw"] > 0)
    ].copy()
    print(f"Active verified motorized vessels in GSA 29 regional register: {len(fleet_clean):,}")

    fleet_clean["upper_co2_t"] = fleet_clean["engine_power_kw"] * UNIFORM_UPPER_EF

    # 2. Isolate Turkish Black Sea fleet (N = 6,584)
    tr_fleet = fleet_clean[fleet_clean["country"] == "Türkiye"].copy()
    tr_fleet["gear_detailed"] = tr_fleet.apply(classify_turkish_gear, axis=1)
    tr_fleet["realistic_co2_t"] = (
        tr_fleet["engine_power_kw"] * tr_fleet["gear_detailed"].map(GEAR_EF_REALISTIC)
    )
    print(f"Active verified Turkish GSA 29 vessels: {len(tr_fleet):,}")

    tr_gear_summary = (
        tr_fleet.groupby("gear_detailed")
        .agg(
            Vessel_Count=("vessel_registration_number", "count"),
            Total_KW=("engine_power_kw", "sum"),
            Mean_KW=("engine_power_kw", "mean"),
            Median_KW=("engine_power_kw", "median"),
            Total_GT=("tonnage_gt", "sum"),
            Mean_GT=("tonnage_gt", "mean"),
            Median_GT=("tonnage_gt", "median"),
            Mean_LOA=("loa_m", "mean"),
            Mean_EII=("eii", "mean"),
            Median_EII=("eii", "median"),
            Mean_Age=("vessel_age", "mean"),
            Vessels_Over_35=("vessel_age", lambda x: int((x > 35).sum())),
            UpperBound_CO2_kt=("upper_co2_t", lambda x: x.sum() / 1000.0),
            Realistic_CO2_kt=("realistic_co2_t", lambda x: round(x.sum() / 1000.0, 2)),
        )
        .reset_index()
    )

    tr_gear_summary["Pct_Fleet"] = (tr_gear_summary["Vessel_Count"] / len(tr_fleet)) * 100.0
    tr_gear_summary["Pct_Over_35"] = (
        tr_gear_summary["Vessels_Over_35"] / tr_gear_summary["Vessel_Count"]
    ) * 100.0
    col_order = [
        "gear_detailed", "Vessel_Count", "Total_KW", "Mean_KW", "Median_KW",
        "Total_GT", "Mean_GT", "Median_GT", "Mean_LOA", "Mean_EII", "Median_EII",
        "Mean_Age", "Vessels_Over_35", "UpperBound_CO2_kt", "Pct_Fleet",
        "Pct_Over_35", "Realistic_CO2_kt",
    ]
    tr_gear_summary = tr_gear_summary[col_order].sort_values(by="Total_KW", ascending=False)
    tr_gear_summary.to_csv(
        os.path.join(TABLES_DIR, "turkish_fleet_segment_architecture.csv"), index=False
    )
    print("\nSaved -> tables/turkish_fleet_segment_architecture.csv")
    print(
        tr_gear_summary[
            ["gear_detailed", "Vessel_Count", "Mean_KW", "Mean_GT", "Mean_EII", "Realistic_CO2_kt", "UpperBound_CO2_kt"]
        ].to_string(index=False)
    )

    # 3. Regional comparison table across Black Sea riparian states
    regional_summary = (
        fleet_clean.groupby("country")
        .agg(
            Vessel_Count=("vessel_registration_number", "count"),
            Total_KW=("engine_power_kw", "sum"),
            Mean_KW=("engine_power_kw", "mean"),
            Total_GT=("tonnage_gt", "sum"),
            Mean_GT=("tonnage_gt", "mean"),
            Mean_EII=("eii", "mean"),
            Median_EII=("eii", "median"),
            Mean_Age=("vessel_age", "mean"),
            Vessels_Over_35=("vessel_age", lambda x: int((x > 35).sum())),
            UpperBound_CO2_kt=("upper_co2_t", lambda x: x.sum() / 1000.0),
        )
        .reset_index()
    )
    regional_summary["Pct_Over_35"] = (
        (regional_summary["Vessels_Over_35"] / regional_summary["Vessel_Count"]) * 100.0
    ).round(2)
    regional_summary["Share_Regional_KW_%"] = (
        regional_summary["Total_KW"] / regional_summary["Total_KW"].sum()
    ) * 100.0
    regional_summary.to_csv(
        os.path.join(TABLES_DIR, "blacksea_regional_fleet_comparison.csv"), index=False
    )
    print("\nSaved -> tables/blacksea_regional_fleet_comparison.csv")

    # 4. Turkish Fleet Economic & Operational Cost Accounts (2000-2024)
    tr_bench = pd.read_csv(os.path.join(DATA_DIR, "turkish_fleet_profitability.csv"))
    tr_bench["Net_Operating_Margin_Depreciated_Pct"] = tr_bench["TR_Op_Margin_Pct"] - 15.0
    tr_2023 = tr_bench[tr_bench["Year"] == 2023].iloc[0]

    real_co2_total = tr_fleet["realistic_co2_t"].sum() / 1000.0
    upper_co2_total = tr_fleet["upper_co2_t"].sum() / 1000.0

    cost_summary = {
        "Metric": [
            "Mean Gross Revenue (Million TRY, 2000-2024 nominal)",
            "Mean Fuel Share (% of Revenue)",
            "Peak Fuel Share (% of Revenue)",
            "Minimum Fuel Share (% of Revenue)",
            "Mean Crew Labor Remuneration Share (% of Revenue)",
            "Mean Gross Cash Operating Margin (EBITDA proxy, %)",
            "Mean Net Operating Margin (Depreciated EBIT proxy, %)",
            "2023 Baseline Gross / Depreciated Net Margin (15% Depr., %)",
            "Total Certified Fleet Power (kW)",
            "Total Fleet Tonnage (GT)",
            "Average Vessel Age (Years)",
            "Vintage Capital Share (>35 Years, % of Vessels)",
            "Realistic Gear-Duty-Cycle Baseline CO2 (kt/yr)",
            "Uniform Industrial Upper-Bound CO2 (kt/yr)",
        ],
        "Turkish Black Sea Fleet Value": [
            f"{tr_bench['TR_Revenue_TRY'].mean() / 1e6:,.2f} Million TRY",
            f"{tr_bench['TR_Fuel_Share_Pct'].mean():.2f}%",
            f"{tr_bench['TR_Fuel_Share_Pct'].max():.2f}% ({int(tr_bench.loc[tr_bench['TR_Fuel_Share_Pct'].idxmax(), 'Year'])})",
            f"{tr_bench['TR_Fuel_Share_Pct'].min():.2f}% ({int(tr_bench.loc[tr_bench['TR_Fuel_Share_Pct'].idxmin(), 'Year'])})",
            f"{tr_bench['TR_Labor_Share_Pct'].mean():.2f}%",
            f"{tr_bench['TR_Op_Margin_Pct'].mean():.2f}%",
            f"{tr_bench['Net_Operating_Margin_Depreciated_Pct'].mean():.2f}%",
            f"{tr_2023['TR_Op_Margin_Pct']:.2f}% / {tr_2023['TR_Op_Margin_Pct'] - 15.0:.2f}%",
            f"{tr_fleet['engine_power_kw'].sum():,.0f} kW",
            f"{tr_fleet['tonnage_gt'].sum():,.0f} GT",
            f"{tr_fleet['vessel_age'].mean():.2f} years",
            f"{(tr_fleet['vessel_age'] > 35).sum() / len(tr_fleet) * 100.0:.2f}%",
            f"{real_co2_total:.2f} kt/yr",
            f"{upper_co2_total:.2f} kt/yr",
        ],
    }

    df_cost_summary = pd.DataFrame(cost_summary)
    df_cost_summary.to_csv(
        os.path.join(TABLES_DIR, "turkish_operational_cost_summary.csv"), index=False
    )
    print("\nSaved -> tables/turkish_operational_cost_summary.csv")
    print(df_cost_summary.to_string(index=False))
    print("\n[SUCCESS] Step 01 completed.")


if __name__ == "__main__":
    main()
