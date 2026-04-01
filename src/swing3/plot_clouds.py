import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from nc.loader import PROJECT_ROOT
from swing3.clouds import (
    MODELS,
    load_low_cloud_clim,
    load_low_cloud_clim_t42,
)
from swing3.plot_mcao_pe_map import setup_map

PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/remote/swing3/plots/clouds")


def _plot_cloud_clim(loader, title: str, out_filename: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    all_data = {}
    for model in MODELS:
        print(f"Loading {model} cloud...")
        clim = loader(model)
        all_data[model] = clim
        print(
            f"\t{model}: shape={clim.shape}, "
            f"range=[{float(np.nanmin(clim.values)):.1f}, {float(np.nanmax(clim.values)):.1f}]%"
        )

    n_cols = 2
    n_rows = (len(MODELS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 6, n_rows * 2.75),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.suptitle(title, fontsize=18)

    im_last = None
    for ax, model in zip(axes.ravel(), MODELS):
        da = all_data[model]
        setup_map(ax)
        im_last = ax.pcolormesh(
            da.coords["lon"].values,
            da.coords["lat"].values,
            da.values,
            transform=ccrs.PlateCarree(),
            cmap="inferno",
        )
        ax.set_title(model, fontsize=14)

    for ax in axes.ravel()[len(MODELS) :]:
        ax.set_visible(False)

    fig.subplots_adjust(
        left=0.14, right=0.88, top=0.95, bottom=0.10, wspace=0.2, hspace=0.05
    )
    cbar_ax = fig.add_axes([0.20, 0.05, 0.60, 0.02])
    fig.colorbar(im_last, cax=cbar_ax, orientation="horizontal").set_label(
        "Low cloud fraction (%)", fontsize=14
    )

    out_path = os.path.join(PLOTS_DIR, out_filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    _plot_cloud_clim(
        load_low_cloud_clim,
        "WisoMIP Mean Low Cloud Fraction (Jan–Apr, 1979–2023)",
        "low_cloud_clim_map.png",
    )
    _plot_cloud_clim(
        load_low_cloud_clim_t42,
        "WisoMIP Mean Low Cloud Fraction on T42 Grid (Jan–Apr, 1979–2023)",
        "low_cloud_clim_map_t42.png",
    )


if __name__ == "__main__":
    main()
