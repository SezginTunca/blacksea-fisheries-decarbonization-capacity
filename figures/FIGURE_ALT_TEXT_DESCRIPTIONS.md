# Figure Accessibility & Alt-Text Descriptions (`figures/`)

**Manuscript:** *Decarbonizing Turkish Black Sea Commercial Fisheries Requires Pairing Industrial Engine-Power Limits with Capacity Reduction*  
**Journal:** *Turkish Journal of Maritime and Marine Sciences* (TrJMMS)  
**Author:** Sezgin Tunca, Ph.D. (ORCID: `0000-0001-6840-1466`)

All figures are rendered at **600 DPI** in both raster (`.png`) and vector (`.pdf`) formats by `scripts/08_generate_publication_figures.py` and `scripts/07_extract_blacksea_ais_validation.py`.

---

## Main-Text Figures

- **`fig1_methodological_workflow.png` / `.pdf` (Figure 1):**  
  *Integrated Empirical, Engineering, and Policy Workflow Framework.* Seven-stage analytical architecture separating directly observed registry, catch, and cost accounts (Phase 1: 1A–1D) from semi-empirical Holtrop-Mennen naval resistance modeling (Phase 2), 34-port spatial autocorrelation diagnostics (Phase 3), 21-cohort directional eco-efficiency modeling (Phase 4), gear-differentiated operational $\text{CO}_2$ inventories and conditional Scenarios A–E (Phase 5), multi-basin panel volatility regressions (Phase 6), and fisheries governance synthesis (Phase 7).

- **`fig2_study_area_map.png` / `.pdf` (Figure 2):**  
  *Master Study Area Map, GSA 29 Zonation, and Fleet Capacity Dashboard.* (a) Bathymetric and jurisdictional map of the Black Sea (GFCM GSA 29) showing the $200\text{ m}$ shelf isobath, Eastern and Western Black Sea operational zones, 27 Black Sea coastal ports (circles), and the Bandırma Sea of Marmara interface hub (diamond) scaled by certified main engine power ($\text{MW}$); (b) Capacity across 21 homogeneous Black Sea cohorts ($n = 4,772$; $\ge 6\text{ m}$); (c) Top 12 registration ports by installed engine power ($\text{MW}$) and vessel count.

- **`fig3_hydrodynamic_froude_scaling.png` / `.pdf` (Figure 3):**  
  *Naval Architecture Scaling, Gear-Specific Benchmarks, and Semi-Empirical Holtrop-Mennen Resistance Regimes.* (a) Within-gear log-log OLS scaling of Engine Intensity Index ($\ln\text{EII}$) against vessel length overall ($\ln\text{LOA}$) across $N = 6,584$ vessels; (b) Empirical kernel density distributions of free-running transit Froude numbers ($Fn$) across all $6,584$ registered vessels; (c) Share of vessels exceeding gear-specific propulsion benchmarks ($\text{PS} > 6$, $\text{OTB/DRB} > 10$, $\text{TBB/MIS} > 12\text{ kW GT}^{-1}$) compared against a uniform $>10\text{ kW GT}^{-1}$ cutoff; (d) Theoretical calm-water wave-making resistance share ($R_W / R_T$, $\%$) modeled via Holtrop-Mennen and ITTC-1957 formulations during free-running transit ($58.24\%\text{--}79.44\%$) versus low-speed active towing and net setting ($2.0\text{--}3.0\text{ knots}$: $<1.3\%$).

- **`fig4_comparative_blacksea_dynamics.png` / `.pdf` (Figure 4):**  
  *Regional Area 37 Fisheries Production, GSA 29 Accounting, and Economic Context.* (a) Historical marine capture landings across FAO Major Area 37 by Black Sea coastal states and the former Soviet Union ($1950\text{--}2024$, cumulative $41,861.36\text{ kt}$; Türkiye 1988 Area 37 peak $621.41\text{ kt}$); (b) Contemporary decade ($2015\text{--}2024$) mean annual Area 37 national landings ($6$-country sum $514.04\text{ kt yr}^{-1}$, Türkiye $340.79\text{ kt yr}^{-1}$ across all Turkish seas) contrasted with strictly GSA 29 Black Sea landings from TÜİK ($256.74\text{ kt yr}^{-1}$, dashed marker); (c) Turkish Black Sea landed harvest value ($2000\text{--}2024$, Million USD via official TCMB exchange rates) with national aquatic trade inset; (d) Cumulative dominant species harvest shares in Area 37 ($1950\text{--}2024$).

- **`fig5_turkish_port_bubble_maps.png` / `.pdf` (Figure 5):**  
  *Port-Level Spatial Distribution of Capacity and Dual Moran's $I$ Diagnostics.* Four-panel cartographic comparison across the 27 Black Sea coastal ports plus the Bandırma Marmara interface hub: (a) Registered commercial vessel counts ($n$); (b) Aggregate enclosed gross tonnage ($\text{GT}$); (c) Total installed main engine power ($\text{MW}$; Global Moran's $I = -0.0691$, $p = 0.4731$ across 27 Black Sea ports); (d) Median Engine Intensity Index ($\text{kW GT}^{-1}$; Global Moran's $I = +0.6408$, $p = 0.0001$).

- **`fig6_decarbonisation_tradeoffs.png` / `.pdf` (Figure 6):**  
  *Conditional Decarbonization Scenarios, 2023 Fuel Shock Elasticities, and Multi-Basin Portfolio Associations.* (a) Annual fleet $\text{CO}_2$ emissions under the primary gear-differentiated duty-cycle inventory ($366.98\text{ kt yr}^{-1}$ base; $-29.31\%$ to $259.43\text{ kt}$ in Scenario E) with uniform capacity upper-bound annotation ($784.92 \rightarrow 617.97\text{ kt}$); (b) Deterministic 2023 TÜİK gross cash ($44.52\%$) and depreciated net ($29.52\%$ at $\delta = 15\%$, shaded $\delta = 8\%\text{--}20\%$ band) operating margin trajectories under real diesel price shocks ($+142.56\%$ zero-net-margin break-even); (c) Surviving-fleet Monte Carlo net operating margin densities ($N = 10,000$) contrasting $0\%$ catch rebound ($\text{B}_0, \text{E}_0$) against conditional $+5\%$ crowding relief ($\text{B}, \text{E}$); (d) Empirical panel scatter and Model 3 partial associations between rolling 5-year Shannon catch diversity ($H'$) and revenue volatility across P10, mean ($\ln\bar{V} = 7.98$), and P90 active fleet scales ($N = 105$).

- **`fig7_dea_eco_efficiency_frontiers.png` / `.pdf` (Figure 7):**  
  *Cohort Data Envelopment Analysis (DEA): Technical Efficiency, Directional Eco-Efficiency, and Abatement Slack.* (a) Distribution of the 21 homogeneous East/West Black Sea cohorts ($n = 4,772$ vessels, $582.51\text{ MW}$) in certified power versus mean guild catch space, sized by cohort GT and colored by directional eco-efficiency ($\text{EE}$); (b) Comparison of VRS technical efficiency ($11/21$ frontier), CRS technical efficiency ($10/21$ frontier), and directional eco-efficiency across all 21 cohorts; (c) Cohort primary duty-cycle $\text{CO}_2$ emissions ($305.59\text{ kt yr}^{-1}$) versus directional technical abatement slack ($30.37\text{ kt CO}_2\text{ yr}^{-1}$, $9.94\%$).

---

## Supplementary Figures

- **`figS1_supplementary_diagnostics.png` / `.pdf` (Supplementary Figure S1):**  
  *Fleet Age Structure, Empirical Moran Scatter Plot, and Parametric $\text{CO}_2$ Inventory Sensitivity.* (a) Age distribution of registered Turkish GSA 29 commercial fishing vessels ($N = 6,584$, mean $22.59\text{ years}$, $9.99\%$ $>35\text{ years}$); (b) Vessel age versus Engine Intensity Index ($\text{kW GT}^{-1}$); (c) Empirical Moran scatter plot of standardized port median EII ($z_i$) against row-standardized spatial lag ($\sum_j w_{ij}z_j$) across the 27 Black Sea coastal ports ($I = +0.6408$, $z = 5.65$, $p = 0.0001$); (d) Parametric sensitivity of annual fleet $\text{CO}_2$ emissions across uniform operating hours ($600\text{--}2,000\text{ h yr}^{-1}$) and effective load multipliers versus the primary gear-differentiated duty-cycle baseline ($366.98\text{ kt CO}_2\text{ yr}^{-1}$).

- **`figS2_ais_speed_validation.png` / `.pdf` (Supplementary Figure S2):**  
  *Illustrative Satellite AIS Fishing-Event Speed Verification (Global Fishing Watch, 2018–2024).* (a) Empirical probability density of apparent fishing-event speed over ground across 849 events from 20 AIS-tracked commercial vessels in the Black Sea; (b) Offshore distance from the mainland coastline ($\text{km}$) versus active fishing speed over ground ($\text{knots}$), showing the 3-nautical-mile ($5.56\text{ km}$) coastal trawl boundary.
