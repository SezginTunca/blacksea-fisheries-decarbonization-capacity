#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT: 02_compile_fao_catch_benchmarks.py
STUDY:  Decarbonizing Turkish Black Sea Commercial Fisheries Requires Pairing
        Industrial Engine-Power Limits with Capacity Reduction
JOURNAL: Turkish Journal of Maritime and Marine Sciences
AUTHOR: Sezgin Tunca, Ph.D. (ORCID: 0000-0001-6840-1466)
PURPOSE: Extract and analyze historical marine capture fisheries landings
         (1950-2024) across FAO Major Fishing Area 37 (Mediterranean and Black
         Sea) for Black Sea riparian states (Türkiye, Bulgaria, Georgia,
         Romania, Russian Federation, Ukraine, and former Soviet Union), and
         compile longitudinal Turkish Area 37 Fuel Use Intensity (FUI) dynamics.
INPUTS:
  - FAO FishStatJ Capture_2026.1.0 database via FAO_CAPTURE_DIR environment
    variable (or pre-extracted data/fao_blacksea_catches_1950_2024.csv)
  - data/turkish_fleet_profitability.csv
OUTPUTS:
  - data/fao_blacksea_catches_1950_2024.csv
  - tables/fao_blacksea_country_annual_pivot.csv
  - tables/fao_blacksea_country_summary.csv
  - tables/fao_blacksea_species_summary.csv
  - tables/turkish_fleet_fui_dynamics.csv
================================================================================
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TABLES_DIR = os.path.join(BASE_DIR, "tables")
FAO_DIR = os.environ.get("FAO_CAPTURE_DIR", os.path.join(DATA_DIR, "FAO_Capture"))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

BLACKSEA_UN = {
    792: "Türkiye",
    100: "Bulgaria",
    268: "Georgia",
    642: "Romania",
    643: "Russian Federation",
    804: "Ukraine",
    810: "Soviet Union [former]",
}


def main():
    print("=" * 80)
    print("STEP 02: Extracting FAO Area 37 Capture Benchmarks & Longitudinal FUI (1950-2024)")
    print("=" * 80)

    local_country_csv = os.path.join(DATA_DIR, "fao_blacksea_catches_1950_2024.csv")

    if os.path.isdir(FAO_DIR) and os.path.exists(os.path.join(FAO_DIR, "Capture_Quantity.csv")):
        cg = pd.read_csv(os.path.join(FAO_DIR, "CL_FI_COUNTRY_GROUPS.csv"))
        sp = pd.read_csv(os.path.join(FAO_DIR, "CL_FI_SPECIES_GROUPS.csv"))
        iso_map = dict(zip(cg["UN_Code"], cg["ISO3_Code"]))
        sp_name_map = dict(zip(sp["3A_Code"], sp["Name_En"]))
        sp_sci_map = dict(zip(sp["3A_Code"], sp["Scientific_Name"]))

        print("Loading FAO Capture_Quantity.csv from primary database...")
        captures = pd.read_csv(os.path.join(FAO_DIR, "Capture_Quantity.csv"))

        area37_mask = captures["AREA.CODE"].astype(str).str.startswith("37")
        country_mask = captures["COUNTRY.UN_CODE"].isin(BLACKSEA_UN.keys())
        bs_catches = captures[area37_mask & country_mask].copy()

        bs_catches["Country"] = bs_catches["COUNTRY.UN_CODE"].map(BLACKSEA_UN)
        bs_catches["ISO3"] = bs_catches["COUNTRY.UN_CODE"].map(iso_map)
        bs_catches["Species_Common"] = (
            bs_catches["SPECIES.ALPHA_3_CODE"]
            .map(sp_name_map)
            .fillna(bs_catches["SPECIES.ALPHA_3_CODE"])
        )
        bs_catches["Species_Scientific"] = (
            bs_catches["SPECIES.ALPHA_3_CODE"].map(sp_sci_map).fillna("")
        )
        bs_catches["Year"] = bs_catches["PERIOD"].astype(int)
        bs_catches["Catch_t"] = bs_catches["VALUE"].fillna(0.0)

        print(
            f"Extracted {len(bs_catches):,} Area 37 catch records "
            f"({bs_catches['Year'].min()}-{bs_catches['Year'].max()})."
        )

        # Country-level annual totals (1950-2024)
        country_annual = (
            bs_catches.groupby(["Country", "ISO3", "Year"])["Catch_t"]
            .sum()
            .reset_index()
        )
        country_annual.to_csv(local_country_csv, index=False)

        # Country summary table (1950-2024 and contemporary 2015-2024 decade)
        summary_rows = []
        for c_code, c_name in BLACKSEA_UN.items():
            c_df = bs_catches[bs_catches["Country"] == c_name]
            if len(c_df) == 0:
                continue
            tot_catch = c_df["Catch_t"].sum()
            peak_rec = c_df.groupby("Year")["Catch_t"].sum().sort_values(ascending=False)
            peak_yr = int(peak_rec.index[0])
            peak_val = float(peak_rec.iloc[0])

            c_recent = c_df[c_df["Year"] >= 2015]
            recent_mean = (
                c_recent.groupby("Year")["Catch_t"].sum().mean()
                if len(c_recent) > 0
                else 0.0
            )
            top_sp = (
                c_df.groupby("Species_Common")["Catch_t"]
                .sum()
                .sort_values(ascending=False)
            )
            dom_sp = (
                f"{top_sp.index[0]} ({top_sp.iloc[0] / tot_catch * 100:.1f}%)"
                if len(top_sp) > 0
                else "N/A"
            )
            summary_rows.append(
                {
                    "Country": c_name,
                    "ISO3": iso_map.get(c_code, ""),
                    "Total_Catch_1950_2024_kt": tot_catch / 1000.0,
                    "Share_Total_%": (tot_catch / bs_catches["Catch_t"].sum()) * 100.0,
                    "Peak_Year": peak_yr,
                    "Peak_Catch_kt": peak_val / 1000.0,
                    "Mean_Catch_2015_2024_kt": recent_mean / 1000.0,
                    "Dominant_Species": dom_sp,
                }
            )

        df_country_summary = pd.DataFrame(summary_rows).sort_values(
            by="Total_Catch_1950_2024_kt", ascending=False
        )
        df_country_summary.to_csv(
            os.path.join(TABLES_DIR, "fao_blacksea_country_summary.csv"), index=False
        )
        print("\nSaved -> tables/fao_blacksea_country_summary.csv")

        # Species harvest breakdown
        species_annual = (
            bs_catches.groupby(["Species_Common", "Species_Scientific"])["Catch_t"]
            .sum()
            .reset_index()
        )
        species_annual["Share_Regional_%"] = (
            species_annual["Catch_t"] / bs_catches["Catch_t"].sum()
        ) * 100.0
        species_annual = species_annual.sort_values(by="Catch_t", ascending=False)

        tr_sp = (
            bs_catches[bs_catches["Country"] == "Türkiye"]
            .groupby("Species_Common")["Catch_t"]
            .sum()
            .to_dict()
        )
        species_annual["Turkiye_Catch_kt"] = (
            species_annual["Species_Common"].map(tr_sp).fillna(0.0) / 1000.0
        )
        species_annual["Turkiye_Share_of_Species_%"] = (
            species_annual["Turkiye_Catch_kt"] * 1000.0 / species_annual["Catch_t"]
        ) * 100.0
        species_annual["Total_Catch_kt"] = species_annual["Catch_t"] / 1000.0

        top_species_table = species_annual.head(12)[
            [
                "Species_Common",
                "Species_Scientific",
                "Total_Catch_kt",
                "Share_Regional_%",
                "Turkiye_Catch_kt",
                "Turkiye_Share_of_Species_%",
            ]
        ]
        top_species_table.to_csv(
            os.path.join(TABLES_DIR, "fao_blacksea_species_summary.csv"), index=False
        )
        print("Saved -> tables/fao_blacksea_species_summary.csv")
    else:
        print("Using pre-extracted FAO Area 37 series in data/fao_blacksea_catches_1950_2024.csv")
        country_annual = pd.read_csv(local_country_csv)

    # Annual pivot table across riparian countries
    country_pivot = country_annual.pivot(
        index="Year", columns="Country", values="Catch_t"
    ).fillna(0.0)
    country_pivot["Total_BlackSea_Area37"] = country_pivot.sum(axis=1)
    country_pivot["Turkiye_Share_%"] = (
        country_pivot["Türkiye"] / country_pivot["Total_BlackSea_Area37"]
    ) * 100.0
    country_pivot.to_csv(
        os.path.join(TABLES_DIR, "fao_blacksea_country_annual_pivot.csv")
    )

    # Longitudinal Area 37 Fuel Use Intensity (FUI) index for Türkiye (2000-2024)
    # Note: Catch_t in FAO Area 37 covers all Turkish marine basins (~420.5 kt in 2023 vs.
    # 256.7 kt decadal GSA 29 Black Sea mean), paired with a 150.0M L national Area 37
    # reference diesel volume (vs. 137.00M L for GSA 29 Black Sea alone in Script 03).
    tr_annual_catch = (
        country_annual[
            (country_annual["Country"] == "Türkiye") & (country_annual["Year"] >= 2000)
        ]
        .groupby("Year")["Catch_t"]
        .sum()
        .reset_index()
    )
    tr_costs = pd.read_csv(os.path.join(DATA_DIR, "turkish_fleet_profitability.csv"))
    fui_df = tr_annual_catch.merge(tr_costs, on="Year")

    fui_df["Estimated_Fuel_Liters"] = (
        fui_df["TR_Fuel_Share_Pct"] / fui_df["TR_Fuel_Share_Pct"].mean() * 150_000_000.0
    )
    fui_df["FUI_Liters_per_Tonne"] = fui_df["Estimated_Fuel_Liters"] / fui_df["Catch_t"]
    fui_df["Harvest_per_KW_kg"] = (fui_df["Catch_t"] * 1000.0) / 675_723.0

    fui_summary = fui_df[
        [
            "Year",
            "Catch_t",
            "TR_Fuel_Share_Pct",
            "TR_Op_Margin_Pct",
            "FUI_Liters_per_Tonne",
            "Harvest_per_KW_kg",
        ]
    ]
    fui_summary.to_csv(
        os.path.join(TABLES_DIR, "turkish_fleet_fui_dynamics.csv"), index=False
    )
    print("Saved -> tables/turkish_fleet_fui_dynamics.csv")
    print("\n[SUCCESS] Step 02 completed.")


if __name__ == "__main__":
    main()
