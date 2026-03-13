import os

import matplotlib.pyplot as plt
import numpy as np

from ds_638_021.potential_temperature import compute_theta_850
from nc.flights import MARLI_FILES
from nc.loader import PROJECT_ROOT
from nc.vars import DS_638_021 as v

PLOTS_DIR = os.path.join(
    PROJECT_ROOT, f"output/{v.dataset}/plots/potential_temperature"
)


def plot_theta_850(
    flight: str,
    result: dict,
    theta_lim: tuple,
    alt_lim: tuple,
) -> str:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, f"{flight.lower()}_theta850.png")

    time_regrid = result["time_regrid_utc_hours"]
    theta = result["theta_850"]
    time_native = result["time_utc_hours"]
    alt = result["altitude"]
    is_interpolated = result["is_interpolated"]

    fig, ax = plt.subplots(figsize=(10, 5))

    measured_mask = ~is_interpolated
    ax.plot(
        time_regrid[measured_mask],
        theta[measured_mask],
        marker="o",
        markersize=3,
        linewidth=1,
        color="tab:blue",
        label=f"$\\theta_{{850}}${' (measured)' if np.any(is_interpolated) else ''}",
    )

    if np.any(is_interpolated):
        ax.plot(
            time_regrid[is_interpolated],
            theta[is_interpolated],
            marker="o",
            markersize=3,
            linewidth=0,
            color="tab:green",
            alpha=0.6,
            label="$\\theta_{850}$ (interpolated)",
        )

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda h, _: f"{int(h):02d}:{int((h % 1) * 60):02d}")
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("$\\theta_{850}$ (K)")
    ax.set_ylim(theta_lim)

    ax2 = ax.twinx()
    ax2.plot(time_native, alt, color="black", linewidth=1.0, label="Aircraft altitude")

    ax2.set_ylabel("Altitude (km)")
    ax2.set_ylim(alt_lim)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=9)

    ax.set_title(
        f"{flight}: Potential Temperature at ~850 hPa "
        f"(H={result['h_850']:.3f} km, p={result['p_850']:.1f} hPa)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def main():
    flights = list(MARLI_FILES.keys())

    results = {}
    for flight in flights:
        results[flight] = compute_theta_850(flight, interpolate=True)

    all_theta = np.concatenate([r["theta_850"] for r in results.values()])
    all_alt = np.concatenate([r["altitude"] for r in results.values()])
    theta_lim = (np.nanmin(all_theta), np.nanmax(all_theta))
    alt_lim = (np.nanmin(all_alt), np.nanmax(all_alt))

    for flight in flights:
        result = results[flight]
        measured = np.count_nonzero(~result["is_interpolated"])
        interpolated = np.count_nonzero(result["is_interpolated"])
        total = len(result["theta_850"])
        plot = plot_theta_850(flight, result, theta_lim, alt_lim)
        print(
            f"{flight}: theta_850 at H={result['h_850']:.4f} km "
            f"(p={result['p_850']:.1f} hPa), "
            f"{measured} measured, {interpolated} interpolated ({total} total) -> {plot}"
        )


if __name__ == "__main__":
    main()
