"""Visualization for the temporal holdout prediction model.

Reads results from run_temporal_analysis and generates:
  - Predicted vs actual scatter (test period 2013-2021)
  - Residuals by year (temporal drift diagnostic)
  - Summary table
"""

import os

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from nc.loader import PROJECT_ROOT
from swing3.config import MODELS
from swing3.diagnostics import _model_coords
from swing3.features import _load_model_arrays
from swing3.predict import PREDICT_PLOTS_DIR
from swing3.plot_mcao_pe_map import setup_map
from swing3.rules._io import load
from swing3.types import TemporalResult

_TEST_YEAR_MIN = 2013
_TEST_YEAR_MAX = 2021
_YEAR_CMAP = plt.cm.plasma


def _year_colors(years: np.ndarray) -> np.ndarray:
    norm = (years - _TEST_YEAR_MIN) / max(_TEST_YEAR_MAX - _TEST_YEAR_MIN, 1)
    return _YEAR_CMAP(norm)


def plot_pred_vs_actual(all_results: dict[str, TemporalResult]) -> None:
    """Scatter of predicted vs actual PE on test set, one panel per model.

    Dots colored by year. 1:1 line shown. Title includes R2 +- std.
    """
    models = [m for m in MODELS if m in all_results]
    n = len(models)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    all_vals = np.concatenate(
        [
            arr
            for m in models
            for arr in (all_results[m]["y_true_test"], all_results[m]["y_pred_test"])
        ]
    )
    vmin, vmax = np.nanpercentile(all_vals, [1, 99])

    for ax, model_name in zip(axes.flat, models):
        r = all_results[model_name]
        colors = _year_colors(r["years_test"])
        ax.scatter(
            r["y_true_test"],
            r["y_pred_test"],
            c=colors,
            s=20,
            alpha=0.4,
            rasterized=True,
        )
        ax.plot([vmin, vmax], [vmin, vmax], "k-", linewidth=0.8, alpha=0.6)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        r2_str = rf"$R^2 = {r['r2_test_mean']:.3f} \pm {r['r2_test_std']:.3f}$"
        ax.set_title(model_name, fontsize=16, pad=6)
        ax.text(0.05, 0.95, r2_str, transform=ax.transAxes, fontsize=12, va="top")
        ax.set_xlabel("Actual PE (%)", fontsize=14)
        ax.set_ylabel("Predicted PE (%)", fontsize=14)
        ax.tick_params(labelsize=12)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    sm = plt.cm.ScalarMappable(
        cmap=_YEAR_CMAP,
        norm=plt.Normalize(_TEST_YEAR_MIN, _TEST_YEAR_MAX),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02, aspect=30)
    cbar.set_label("Test year", fontsize=14)
    cbar.set_ticks(list(range(_TEST_YEAR_MIN, _TEST_YEAR_MAX + 1)))
    cbar.ax.tick_params(labelsize=12)

    fig.suptitle(
        f"Predicted vs Actual PE (test {_TEST_YEAR_MIN}-{_TEST_YEAR_MAX})",
        fontsize=18,
    )

    out = os.path.join(PREDICT_PLOTS_DIR, "pred_vs_actual.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_residuals_by_year(all_results: dict[str, TemporalResult]) -> None:
    """Box-and-whisker of (y_true - y_pred) per test year, one panel per model."""
    models = [m for m in MODELS if m in all_results]
    n = len(models)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 5 * nrows),
        squeeze=False,
        constrained_layout=True,
        sharey=True,
    )

    for ax, model_name in zip(axes.flat, models):
        r = all_results[model_name]
        residuals = r["y_true_test"] - r["y_pred_test"]
        years = r["years_test"]

        model_years = sorted(np.unique(years).tolist())
        data_by_year = [residuals[years == yr] for yr in model_years]
        ax.boxplot(
            data_by_year,
            tick_labels=model_years,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"facecolor": "steelblue", "alpha": 0.6},
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
            whiskerprops={"linewidth": 0.9},
            capprops={"linewidth": 0.9},
        )
        ax.axhline(0, color="red", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.set_title(model_name, fontsize=16, pad=6)
        ax.set_xlabel("Test Year", fontsize=14)
        ax.tick_params(labelsize=12)

    for ax in axes[:, 0]:
        ax.set_ylabel("Residual PE (%)", fontsize=14)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    fig.suptitle("Residuals by Year", fontsize=18)

    out = os.path.join(PREDICT_PLOTS_DIR, "residuals_by_year.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_summary_table(
    all_results: dict[str, TemporalResult], train_cutoff: int = 2012
) -> None:
    """Print summary R2 table across models.

    r2_test_std reflects model variance across seeds, not test-set sampling variance.
    """
    n_test_years = 2021 - train_cutoff
    n_independent = n_test_years * 4
    print(
        f"\nTest period: {train_cutoff + 1}-2021 ({n_test_years} years, {n_independent} independent JFMA time points)"
    )
    print(
        "Note: +-std reflects model variance across seeds, not test-set sampling uncertainty.\n"
    )
    print(f"{'Model':<10}  {'Features':>8}  {'Test R2':>13}")
    print("-" * 38)
    for model_name in MODELS:
        if model_name not in all_results:
            continue
        r = all_results[model_name]
        print(
            f"{model_name:<10}  {r['n_features']:>8}  "
            f"{r['r2_test_mean']:>6.3f} +-{r['r2_test_std']:.3f}"
        )


# TODO: store lat_idx_test/lon_idx_test in TemporalResult so this function
# doesn't need to re-derive the NaN mask via _load_model_arrays.
def compute_spatial_pe_maps(
    model_name: str, result: TemporalResult, train_cutoff: int = 2012
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map temporal holdout predictions back to mean (lat, lon) arrays.

    Re-derives the NaN filtering mask from _load_model_arrays to map each
    valid test sample's raveled index back to (lat_idx, lon_idx), then
    accumulates to get per-cell means over the test period.

    Returns: (lat_vals, lon_vals, actual_mean, pred_mean, diff_mean)
    """
    arrays, time_groups_full = _load_model_arrays(model_name)
    lat_vals, lon_vals = _model_coords(model_name)
    nlat, nlon = len(lat_vals), len(lon_vals)

    pe_full = arrays["pref"]
    years_full = (1979 + time_groups_full // 4).astype(int)

    # Reproduce the NaN mask from load_predict_features
    nan_ok = {"low_cloud", "omega_925", "pref"}
    mask = np.isfinite(pe_full) & (pe_full >= 0) & (pe_full <= 100)
    for key, arr in arrays.items():
        if key not in nan_ok:
            mask &= np.isfinite(arr)

    valid_indices = np.where(mask)[0]
    years_valid = years_full[mask]
    test_mask_valid = years_valid > train_cutoff
    valid_test_indices = valid_indices[test_mask_valid]

    y_pred = result["y_pred_test"]
    y_true = result["y_true_test"]

    assert len(y_pred) == int(test_mask_valid.sum()), (
        f"[{model_name}] Mismatch: {len(y_pred)} predictions vs "
        f"{test_mask_valid.sum()} test samples"
    )

    # Each raveled index maps to (time, lat, lon) in C order
    spatial_idx = valid_test_indices % (nlat * nlon)
    lat_idx = spatial_idx // nlon
    lon_idx = spatial_idx % nlon

    pred_sum = np.zeros((nlat, nlon))
    true_sum = np.zeros((nlat, nlon))
    count = np.zeros((nlat, nlon), dtype=int)
    np.add.at(pred_sum, (lat_idx, lon_idx), y_pred)
    np.add.at(true_sum, (lat_idx, lon_idx), y_true)
    np.add.at(count, (lat_idx, lon_idx), 1)

    with np.errstate(invalid="ignore"):
        covered = count > 0
        pred_mean = np.where(covered, pred_sum / count, np.nan)
        true_mean = np.where(covered, true_sum / count, np.nan)

    return lat_vals, lon_vals, true_mean, pred_mean, true_mean - pred_mean


def plot_spatial_pe_comparison(all_results: dict[str, TemporalResult]) -> None:
    """Spatial mean PE maps for the test period: actual | predicted | bias.

    One row per model, three columns: actual PE, predicted PE, and
    actual minus predicted. Actual and predicted share a color scale;
    the bias panel uses a symmetric diverging scale.
    """
    models = [m for m in MODELS if m in all_results]

    spatial = {m: compute_spatial_pe_maps(m, all_results[m]) for m in models}

    all_pe = np.concatenate(
        [arr for m in models for arr in (spatial[m][2].ravel(), spatial[m][3].ravel())]
    )
    all_pe = all_pe[np.isfinite(all_pe)]
    pe_norm = mcolors.Normalize(
        vmin=float(np.percentile(all_pe, 2)),
        vmax=float(np.percentile(all_pe, 98)),
    )

    all_diff = np.concatenate([spatial[m][4].ravel() for m in models])
    all_diff = all_diff[np.isfinite(all_diff)]
    diff_lim = float(np.percentile(np.abs(all_diff), 98))
    diff_norm = mcolors.Normalize(vmin=-diff_lim, vmax=diff_lim)

    n_models = len(models)
    n_cols = 3
    fig, axes = plt.subplots(
        n_models,
        n_cols,
        figsize=(n_cols * 6, n_models * 2.75),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )

    col_labels = ["Actual PE (%)", "Predicted PE (%)", "Actual - Predicted (%)"]
    col_cmaps = ["inferno", "inferno", "RdBu_r"]
    col_norms = [pe_norm, pe_norm, diff_norm]

    im_pe = None
    im_diff = None
    for row, model in enumerate(models):
        lat_vals, lon_vals, actual, pred, diff = spatial[model]
        panels = [actual, pred, diff]

        for col, (data, cmap, norm) in enumerate(zip(panels, col_cmaps, col_norms)):
            ax = axes[row, col]
            setup_map(ax)
            im = ax.pcolormesh(
                lon_vals,
                lat_vals,
                data,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
            )
            if col < 2:
                im_pe = im
            else:
                im_diff = im

            if row == 0:
                ax.set_title(col_labels[col], fontsize=13, pad=5, fontweight="bold")
            if col == 0:
                ax.text(
                    -0.22,
                    0.5,
                    model,
                    transform=ax.transAxes,
                    fontsize=12,
                    va="center",
                    ha="right",
                    fontweight="bold",
                )

    fig.subplots_adjust(
        left=0.14, right=0.90, top=0.95, bottom=0.09, wspace=0.18, hspace=0.12
    )

    pe_cbar_ax = fig.add_axes([0.10, 0.045, 0.37, 0.018])
    fig.colorbar(im_pe, cax=pe_cbar_ax, orientation="horizontal").set_label(
        "PE (%)", fontsize=12
    )

    diff_cbar_ax = fig.add_axes([0.55, 0.045, 0.30, 0.018])
    fig.colorbar(im_diff, cax=diff_cbar_ax, orientation="horizontal").set_label(
        "Bias (%)", fontsize=12
    )

    fig.suptitle(
        f"Spatial mean PE -- actual vs predicted (test {_TEST_YEAR_MIN}-{_TEST_YEAR_MAX})",
        fontsize=16,
        y=0.98,
    )

    out = os.path.join(PREDICT_PLOTS_DIR, "spatial_pe_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    os.makedirs(PREDICT_PLOTS_DIR, exist_ok=True)

    all_results: dict[str, TemporalResult] = {}
    for model_name in MODELS:
        pkl = os.path.join(
            PROJECT_ROOT, "output/swing3", model_name, "temporal_prediction.pkl"
        )
        all_results[model_name] = load(pkl)

    print("Generating plots...")
    plot_pred_vs_actual(all_results)
    plot_residuals_by_year(all_results)
    plot_spatial_pe_comparison(all_results)

    print_summary_table(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
