"""Diagnostic plots: time series, geographic climatology, interseasonal variation.

Geographic: one figure per variable (all models as columns).
Time series / interseasonal: one figure per variable group.
"""

import os

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from nc.loader import PROJECT_ROOT
from nc.remote import SWING3_MODELS
from swing3.config import GROUP_LABELS, PREDICTOR_GROUPS
from swing3.diagnostics import DIAG_LABELS, load_diagnostic_spatial
from swing3.plot_mcao_pe_map import setup_map

MODELS = list(SWING3_MODELS.keys())
PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/swing3/plots/diagnostics")

_MODEL_COLORS = plt.cm.tab10(np.linspace(0, 0.9, len(MODELS)))
_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr"]
_DIVERGING_VARS = {"mcao", "omega_925", "omega_700"}

# pe is the target, not in PREDICTOR_GROUPS; treat it as its own group
DIAG_GROUPS: dict[str, list[str]] = {
    "pe": ["pe"],
    **{g: list(vs) for g, vs in PREDICTOR_GROUPS.items()},
}
DIAG_GROUP_LABELS: dict[str, str] = {
    "pe": "Precipitation Efficiency",
    **GROUP_LABELS,
}


def _present(var_names: list[str], all_data: dict) -> list[str]:
    """Keep only vars present in at least one model's data dict."""
    present = set().union(*[set(d.keys()) for d in all_data.values()])
    return [v for v in var_names if v in present]


def plot_geographic_clim(
    all_data: dict[str, dict[str, xr.DataArray]], var: str
) -> None:
    """One figure per variable: 2-column grid matching the cloud clim map style."""
    cmap = "RdBu_r" if var in _DIVERGING_VARS else "inferno"
    label = DIAG_LABELS.get(var, var)

    models_with_var = [m for m in MODELS if var in all_data[m]]
    if not models_with_var:
        return

    clim_arrays = [all_data[m][var].mean(dim="time").values for m in models_with_var]
    all_vals = np.concatenate([c.ravel() for c in clim_arrays])
    all_vals = all_vals[np.isfinite(all_vals)]
    vmin, vmax = float(np.percentile(all_vals, 2)), float(np.percentile(all_vals, 98))
    if var in _DIVERGING_VARS:
        lim = max(abs(vmin), abs(vmax))
        vmin, vmax = -lim, lim
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    n_cols = 2
    n_rows = (len(models_with_var) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 6, n_rows * 2.75),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.suptitle(f"WisoMIP Mean {label} (Jan-Apr)", fontsize=18)

    im_last = None
    for ax, model in zip(axes.ravel(), models_with_var):
        setup_map(ax)
        da = all_data[model][var].mean(dim="time")
        im_last = ax.pcolormesh(
            da["lon"].values,
            da["lat"].values,
            da.values,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
        )
        ax.set_title(model, fontsize=14)

    for ax in axes.ravel()[len(models_with_var) :]:
        ax.set_visible(False)

    fig.subplots_adjust(
        left=0.14, right=0.88, top=0.95, bottom=0.10, wspace=0.2, hspace=0.05
    )
    cbar_ax = fig.add_axes([0.20, 0.05, 0.60, 0.02])
    fig.colorbar(im_last, cax=cbar_ax, orientation="horizontal").set_label(
        label, fontsize=14
    )

    out = os.path.join(PLOTS_DIR, f"geographic_clim_{var}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_time_series(
    all_data: dict[str, dict[str, xr.DataArray]],
    group_name: str,
    var_names: list[str],
) -> None:
    """Annual JFMA spatial-mean time series; one row per variable in the group."""
    n_vars = len(var_names)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 2.5 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    for ax, var in zip(axes, var_names):
        for model, color in zip(MODELS, _MODEL_COLORS):
            if var not in all_data[model]:
                continue
            da = all_data[model][var]
            sm = da.mean(dim=[d for d in da.dims if d != "time"])
            annual = sm.groupby("time.year").mean()
            ax.plot(
                annual["year"].values,
                annual.values,
                color=color,
                label=model,
                linewidth=1.5,
            )
        if var in _DIVERGING_VARS:
            ax.axhline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.45)
        ax.set_ylabel(DIAG_LABELS.get(var, var), fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=9)

    axes[-1].set_xlabel("Year", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(MODELS),
        fontsize=9,
        framealpha=0.85,
        handlelength=1.5,
    )
    group_label = DIAG_GROUP_LABELS.get(group_name, group_name)
    fig.suptitle(
        f"Annual JFMA spatial-mean -- {group_label} (CAESAR domain)",
        fontsize=12,
        y=1.04,
    )
    fig.tight_layout()

    out = os.path.join(PLOTS_DIR, f"time_series_{group_name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_interseasonal(
    all_data: dict[str, dict[str, xr.DataArray]],
    group_name: str,
    var_names: list[str],
) -> None:
    """JFMA spatial-mean +- 1std by month; one panel per variable in the group."""
    n_vars = len(var_names)
    n_cols = min(n_vars, 3)
    n_rows = (n_vars + n_cols - 1) // n_cols
    x = np.arange(4)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.8 * n_rows), squeeze=False
    )

    for idx, var in enumerate(var_names):
        ax = axes[idx // n_cols, idx % n_cols]
        for model, color in zip(MODELS, _MODEL_COLORS):
            if var not in all_data[model]:
                continue
            da = all_data[model][var]
            sm = da.mean(dim=[d for d in da.dims if d != "time"])
            monthly_mean = sm.groupby("time.month").mean().values
            monthly_std = sm.groupby("time.month").std().values
            ax.plot(
                x,
                monthly_mean,
                color=color,
                label=model,
                linewidth=1.8,
                marker="o",
                markersize=5,
                zorder=3,
            )
            ax.fill_between(
                x,
                monthly_mean - monthly_std,
                monthly_mean + monthly_std,
                color=color,
                alpha=0.13,
                zorder=2,
            )
        if var in _DIVERGING_VARS:
            ax.axhline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.45)
        ax.set_xticks(x)
        ax.set_xticklabels(_MONTH_NAMES, fontsize=10)
        ax.set_title(DIAG_LABELS.get(var, var), fontsize=11)
        ax.set_ylabel(DIAG_LABELS.get(var, var), fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")
        ax.tick_params(labelsize=9)

    for idx in range(n_vars, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(MODELS),
        fontsize=9,
        framealpha=0.85,
        handlelength=1.5,
    )
    group_label = DIAG_GROUP_LABELS.get(group_name, group_name)
    fig.suptitle(
        f"Interseasonal variation: JFMA spatial-mean +- 1std -- {group_label} (CAESAR domain)",
        fontsize=12,
        y=1.04,
    )
    fig.tight_layout()

    out = os.path.join(PLOTS_DIR, f"interseasonal_{group_name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading diagnostic data...")
    all_data: dict[str, dict[str, xr.DataArray]] = {}
    for model in MODELS:
        print(f"  {model}...")
        all_data[model] = load_diagnostic_spatial(model)

    print("Plotting geographic climatology (one figure per variable)...")
    for var in _present(list(DIAG_LABELS.keys()), all_data):
        plot_geographic_clim(all_data, var)

    print("Plotting time series and interseasonal (one figure per group)...")
    for group_name, group_vars in DIAG_GROUPS.items():
        vars_in_group = _present(group_vars, all_data)
        if not vars_in_group:
            continue
        plot_time_series(all_data, group_name, vars_in_group)
        plot_interseasonal(all_data, group_name, vars_in_group)

    print("Done.")


if __name__ == "__main__":
    main()
