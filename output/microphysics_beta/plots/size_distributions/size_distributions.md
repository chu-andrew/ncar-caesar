# Particle Size Distributions

Note: All plots here are limited to low-level legs.

## Mean Size Distribution

![](mean_dNdD_vs_D.png)

## Size Distributions Colored by Water Path

| | WVP | LWP |
|-|-----|-----|
| Scatter | ![](dNdD_vs_D_colored_by_WVP.png) | ![](dNdD_vs_D_colored_by_LWP.png) |
| Stratified | ![](dNdD_vs_D_stratified_by_WVP.png) | ![](dNdD_vs_D_stratified_by_LWP.png) |

## Integrated Properties vs Water Path

**$M_k$** — $k$-th moment of the size distribution:

$$M_k = \sum_i D_i^k \cdot \frac{dN}{dD}\bigg|_i \cdot \Delta D_i$$

- $M_0$ (# / $\text{m}^3$): total number concentration
- $M_3$ ($\mu \text{m}^3/\text{m}^3$): volume-weighted moment

**$D_{\text{eff}}$** — effective diameter:

$$D_{\text{eff}} = \frac{M_3}{M_2} \quad (\mu\text{m})$$

| WVP | LWP |
|-----|-----|
| ![](integrated_properties_vs_WVP.png) | ![](integrated_properties_vs_LWP.png) |

## dN/dD Heatmaps by Flight

| Flight | Heatmap |
|--------|---------|
| RF01 | ![](dNdD_heatmap_RF01.png) |
| RF02 | ![](dNdD_heatmap_RF02.png) |
| RF05 | ![](dNdD_heatmap_RF05.png) |
| RF06 | ![](dNdD_heatmap_RF06.png) |
| RF07 | ![](dNdD_heatmap_RF07.png) |
| RF09 | ![](dNdD_heatmap_RF09.png) |
| RF10 | ![](dNdD_heatmap_RF10.png) |
