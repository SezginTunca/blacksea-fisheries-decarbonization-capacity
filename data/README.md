# Data Sources & Institutional Access Documentation

> **Important Note on Data Distribution:**  
> In compliance with institutional data-governance policies, **no raw or processed datasets (`.csv`, `.xlsx`, `.geojson`) are hosted or distributed in this repository or on Zenodo.** All empirical analyses in this study rely on publicly accessible institutional statistical portals and registries listed below. Researchers can obtain the underlying series directly from these official portals and place them in this `data/` directory to execute the analytical scripts in `scripts/`.

---

## 1. Primary Institutional Data Sources

| Dataset / Series | Institutional Provider | Coverage | Access Portal / Reference |
| :--- | :--- | :--- | :--- |
| **GSA 29 Authorized Vessel Register Microdata** | General Fisheries Commission for the Mediterranean (GFCM / FAO) | Black Sea riparian commercial fleets ($N = 9,105$ regional vessels; $N = 6,584$ active Turkish vessels with certified LOA, GT, and kW) | [GFCM Authorized Vessel List (AVL) & Fleet Register](https://www.fao.org/gfcm/data/fleet/avl) |
| **Turkish Commercial Marine Capture Landings, Ex-Vessel Prices & Fleet Accounts** | Turkish Statistical Institute (TÜİK) & Ministry of Agriculture and Forestry, Directorate General of Fisheries and Aquaculture (BSGM) | Annual landings by species and marine basin ($2000\text{--}2024$), active vessel counts, and 2023 fleet operational cost accounts (fuel, crew labor, maintenance) | [TÜİK Fishery Statistics Portal](https://data.tuik.gov.tr/) & [BSGM Fisheries Statistics](https://www.tarimorman.gov.tr/BSGM) |
| **Historical Regional Capture Production (FAO Major Fishing Area 37)** | Food and Agriculture Organization of the United Nations (FAO) | Annual capture landings by country and species in the Mediterranean and Black Sea ($1950\text{--}2024$, FishStatJ Release `2026.1.0`) | [FAO FishStatJ Capture Production Database](https://www.fao.org/fishery/en/statistics/software/fishstatj) |
| **Macroeconomic Exchange Rate & Inflation Deflator Series** | Central Bank of the Republic of Türkiye (TCMB) & TÜİK | Official annual average USD/TRY exchange rate and Consumer Price Index (CPI, $2003 = 100$, $2000\text{--}2024$) | [TCMB Electronic Data Delivery System (EVDS)](https://evds2.tcmb.gov.tr/) |
| **Black Sea Sea-Surface Temperature (SST) & Chlorophyll-$a$ Series** | NOAA Optimum Interpolation SST (OISST v2.1) & Copernicus Marine Service (CMEMS) | Annual basin-level SST anomalies ($^\circ\text{C}$ relative to $1991\text{--}2020$ climatology) and log surface chlorophyll-$a$ ($\text{mg m}^{-3}$, $2000\text{--}2024$) | [NOAA OISST v2.1](https://www.ncei.noaa.gov/products/optimum-interpolation-sst) & [Copernicus Marine Service](https://marine.copernicus.eu/) |
| **Illustrative Satellite AIS Fishing-Event Telemetry** | Global Fishing Watch (GFW) | Apparent fishing-event speed-over-ground records ($849$ events across $20$ AIS-equipped vessels in the Black Sea, $2018\text{--}2024$) | [Global Fishing Watch Data Portal](https://globalfishingwatch.org/data-download/) |
| **Cartographic Basemap Layers** | Natural Earth & GFCM Geographical Subareas | $1:50\text{m}$ coastline, land polygons, administrative boundaries, $200\text{ m}$ shelf bathymetry, and GFCM GSA polygons | [Natural Earth Data](https://www.naturalearthdata.com/) & [GFCM Geographical Subareas](https://www.fao.org/gfcm/data/maps/gsas) |

---

## 2. Expected Local File Schema (`data/`)

When running the pipeline locally after retrieving the institutional series above, place the compiled files in `data/` using the following filenames expected by `scripts/01_compile_fleet_and_economic_data.py` through `scripts/08_generate_publication_figures.py`:

1. `gfcm_blacksea_fleet.csv` — Vessel-level GFCM GSA 29 registry records (`country`, `vessel_registration_number`, `port_of_registration`, `gear1`, `gear_group`, `size_class`, `loa_m`, `tonnage_gt`, `engine_power_kw`, `eii`, `vessel_age`, `ircs_if_applicable`).
2. `turkish_fleet_profitability.csv` — Longitudinal Turkish marine fishery revenue and cost-share accounts (`Year`, `TR_Revenue_TRY`, `TR_Fuel_Share_Pct`, `TR_Labor_Share_Pct`, `TR_Op_Margin_Pct`).
3. `fao_blacksea_catches_1950_2024.csv` — Annual FAO Area 37 capture landings (`Country`, `ISO3`, `Year`, `Catch_t`).
4. `turkish_basin_panel_data.csv` — Five-basin longitudinal panel (`Year`, `Region`, `Total_Value_USD`, `Rev_Roll_CV`, `Catch_Roll_CV`, `Shannon_Roll_Mean`, `Vessels_Roll_Mean`, `USD_TRY_Roll_Mean`, `CPI_Roll_Mean`, `SST_Anomaly`, `Chla_mg_m3`).
5. `gfw_blacksea_ais_events_20vessels.csv` — Illustrative Black Sea AIS fishing-event telemetry subset (`mmsi`, `vessel_name`, `gear_type`, `year`, `avg_speed_knots`, `distance_shore_km`, `distance_port_km`, `longitude`, `latitude`, `basin`).
6. `gsas.geojson`, `ne_50m_land.geojson`, `ne_50m_admin_0_countries.geojson`, `ne_10m_bathymetry_K_200.geojson`, `ne_50m_rivers_lake_centerlines.geojson` — Optional Natural Earth and GFCM vector basemap layers used by `scripts/08_generate_publication_figures.py` for Figures 2 and 5.
