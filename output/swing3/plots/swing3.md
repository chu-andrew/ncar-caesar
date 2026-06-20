# WisoMIP Swing3

## MCAO-PE Relationship

Exploratory characterization of how PE varies with the marine cold-air outbreak (MCAO) index across all models and overlaid atmospheric/isotope fields. These plots motivated the MCAO feature and the hexbin coloring approach used throughout the SHAP dependence plots.

### PE vs. MCAO by model

![PE vs. MCAO hexbin by model](mcao_pe/pe_vs_mcao_hexbin_by_model.png)

### PE distributions by model

![KDE by model](mcao_pe/kde_by_model.png)

### Isotope overlays

| dD precipitation (median) | d-excess precipitation (median) |
|--|--|
| ![](mcao_pe/pe_vs_mcao_hexbin_dDp_median.png) | ![](mcao_pe/pe_vs_mcao_hexbin_dexcessp_median.png) |

| dD vapor 800-925 hPa (mean) | dD vapor 600-800 hPa (mean) |
|--|--|
| ![](mcao_pe/pe_vs_mcao_hexbin_dD_bl_mean.png) | ![](mcao_pe/pe_vs_mcao_hexbin_dD_ft_mean.png) |

### Surface field overlays

Grid cells with evaporation ≤ 0 are excluded before computing ln(pr/ev).

| Specific humidity (median) | Precipitation rate (mean) |
|--|--|
| ![](mcao_pe/pe_vs_mcao_hexbin_sh_median.png) | ![](mcao_pe/pe_vs_mcao_hexbin_pr_mean.png) |

| Evaporation rate (mean) | ln(precipitation/evaporation) (mean) |
|--|--|
| ![](mcao_pe/pe_vs_mcao_hexbin_ev_mean.png) | ![](mcao_pe/pe_vs_mcao_hexbin_pr_over_ev_mean.png) |

---

## Low Cloud Climatology

Time-mean JFMA low cloud fraction over the CAESAR region. The T42 panel shows the field regridded to a coarser resolution for cross-model comparison; the native-resolution panel preserves each model's original grid.

![Low cloud climatology (native resolution)](clouds/low_cloud_clim_map.png)

![Low cloud climatology (T42)](clouds/low_cloud_clim_map_t42.png)
