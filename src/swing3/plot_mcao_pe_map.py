import os
from typing import Literal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

from nc.flights import CAESAR_BOUNDS as bounds
from nc.loader import PROJECT_ROOT
from nc.remote import SWING3_MODELS
from swing3.models import load_mcao_pe_clim
from swing3.sst import load_sst
from nc.vars import SWING3 as v_swing3

PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/remote/swing3/plots")
MODELS = list(SWING3_MODELS.keys())


def setup_map(ax: plt.Axes) -> None:
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=cfeature.COLORS["water"])
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=cfeature.COLORS["land"])
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, linestyle="--")
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.6)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_extent(
        [bounds["MIN_LON"], bounds["MAX_LON"], bounds["MIN_LAT"], bounds["MAX_LAT"]],
        crs=ccrs.PlateCarree(),
    )


def draw_panel(ax, da, title, cmap, vmin, vmax):
    setup_map(ax)
    im = ax.pcolormesh(
        da.coords[v_swing3.lon].values,
        da.coords[v_swing3.lat].values,
        da.values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=14)
    return im


def make_figure(
    field: Literal["mcao", "pe"],
    all_data: dict,
    cmap: str,
    vmin: float,
    vmax: float,
    cb_label: str,
    title: str,
):
    proj = ccrs.PlateCarree()
    n_cols = 2
    n_rows = (len(MODELS) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 6, n_rows * 2.75),
        subplot_kw={"projection": proj},
    )
    fig.suptitle(
        f"WisoMIP Mean {title} (Jan–Apr)",
        fontsize=18,
    )

    im_last = None
    for ax, model in zip(axes.ravel(), MODELS):
        mcao_clim, pe_clim = all_data[model]
        da = mcao_clim if field == "mcao" else pe_clim
        im_last = draw_panel(ax, da, model, cmap, vmin, vmax)

    for ax in axes.ravel()[len(MODELS) :]:
        ax.set_visible(False)

    fig.subplots_adjust(
        left=0.14, right=0.88, top=0.95, bottom=0.10, wspace=0.2, hspace=0.05
    )
    cbar_ax = fig.add_axes([0.20, 0.05, 0.60, 0.02])
    cb = fig.colorbar(im_last, cax=cbar_ax, label=cb_label, orientation="horizontal")
    cb.set_label(cb_label, fontsize=14)

    out_path = os.path.join(PLOTS_DIR, f"{field}_clim_map.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading SST...")
    sst_da = load_sst()

    all_data = {}
    for model in MODELS:
        print(f"Loading {model}...")
        mcao_clim, pe_clim = load_mcao_pe_clim(model, sst_da=sst_da)
        all_data[model] = (mcao_clim, pe_clim)

    # shared color limits from all models (1st–99th percentile)
    all_mcao = np.stack([d[0].values for d in all_data.values()])
    all_pe = np.stack([d[1].values for d in all_data.values()])

    mcao_abs = np.nanpercentile(np.abs(all_mcao), 99)
    mcao_vmin, mcao_vmax = -mcao_abs, mcao_abs

    pe_vmin = np.nanpercentile(all_pe, 1)
    pe_vmax = np.nanpercentile(all_pe, 99)

    make_figure("mcao", all_data, "inferno", mcao_vmin, mcao_vmax, "MCAO (K)", "MCAO")
    make_figure(
        "pe",
        all_data,
        "inferno",
        pe_vmin,
        pe_vmax,
        "PE (%)",
        "Precipitation Efficiency",
    )


if __name__ == "__main__":
    main()
