# WisoMIP

## Climatological Maps

| MCAO                   | PE                   |
|------------------------|----------------------|
| ![](mcao_clim_map.png) | ![](pe_clim_map.png) |

## PE vs MCAO Hexbin by Model

PE truncated at 100% (values > 100% excluded).

![](pe_vs_mcao_hexbin_by_model.png)

## Distributions by Model
PE truncated at 100% (values > 100% excluded).
The x-axis is limited to between the first and 99th percentiles of all observations.

![](kde_by_model.png)

## PE vs MCAO Colored by Overlay Fields

All plots below use PE <= 100% only.

### Isotopes

| dD precipitation (median) | d-excess precipitation (median) |
|--|--|
| ![](pe_vs_mcao_hexbin_dDp_median.png) | ![](pe_vs_mcao_hexbin_dexcessp_median.png) |

| dD vapor 600-800 hPa (mean) | dD vapor 800-925 hPa (mean) |
|--|--|
| ![](pe_vs_mcao_hexbin_dD_ft_mean.png) | ![](pe_vs_mcao_hexbin_dD_bl_mean.png) |

### Surface Fields

For pr/ev: grid cells with evaporation <= 0 are excluded before computing the ratio, so that we can take the natural log (for better color definition).

| Specific humidity (median) | Precipitation rate (mean) |
|--|--|
| ![](pe_vs_mcao_hexbin_sh_median.png) | ![](pe_vs_mcao_hexbin_pr_mean.png) |

| Evaporation rate (mean) | ln(precipitation/evaporation) (mean) |
|--|--|
| ![](pe_vs_mcao_hexbin_ev_mean.png) | ![](pe_vs_mcao_hexbin_pr_over_ev_mean.png) |
