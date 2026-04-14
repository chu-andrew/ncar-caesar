# SHAP Analysis

## Beeswarm plots

Each figure shows one model across four staged predictor sets (Thermo -> +Dynamics -> +Clouds -> +Isotopes). Within each panel, features are ranked by mean |SHAP value| (most important at top). Each dot is one grid cell-month sample. The x-axis shows the SHAP value: how much that feature pushed the predicted PE above (positive) or below (negative) the model baseline. Dot color reflects the feature's value -- red = high, blue = low -- so a cluster of red dots on the right means high feature values increase PE, while red dots on the left mean high values decrease PE. The R^2 in the title is the mean out-of-sample test score across 25 random splits.

### CAM5
![CAM5 beeswarms](CAM5_beeswarms.png)

### CAM6
![CAM6 beeswarms](CAM6_beeswarms.png)

### ECHAM
![ECHAM beeswarms](ECHAM_beeswarms.png)

### GISS
![GISS beeswarms](GISS_beeswarms.png)

### GSM
![GSM beeswarms](GSM_beeswarms.png)

### LMDZ
![LMDZ beeswarms](LMDZ_beeswarms.png)

### MIROC
![MIROC beeswarms](MIROC_beeswarms.png)

---

## Intermodel feature importance heatmap (Stage 4)

Normalized mean |SHAP| per feature across all models.

![Intermodel heatmap](intermodel_heatmap.png)

---

## MCAO SHAP dependence (colored by low cloud fraction)

Each figure shows one stage. Within each panel, the x-axis is the MCAO value and the y-axis is the SHAP value for MCAO -- i.e., how much MCAO alone shifted the predicted PE for that sample. Points above zero indicate MCAO increased PE; points below indicate it decreased PE. In Stages 3-4, dots are colored by low cloud fraction (red = high cloud, blue = low cloud), revealing how the MCAO-PE relationship is affected by cloud cover. Stages 1-2 are shown in gray since low cloud is not yet included as a predictor. Axes are shared across models within each stage to allow direct comparison.

### Stage 1: Thermo
![MCAO dependence -Stage 1](mcao_dependence_s1.png)

### Stage 2: + Dynamics
![MCAO dependence -Stage 2](mcao_dependence_s2.png)

### Stage 3: + Clouds
![MCAO dependence -Stage 3](mcao_dependence_s3.png)

### Stage 4: + Isotopes
![MCAO dependence -Stage 4](mcao_dependence_s4.png)
