import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from ds_638_021.potential_temperature import compute_theta_850
from nc.flights import VERTICAL_LEGS
from nc.loader import PROJECT_ROOT, open_dataset
from nc.units import m_to_km
from nc.vars import DS_638_001 as v001
from nc.vars import DS_638_021 as v

PLOTS_DIR = os.path.join(
    PROJECT_ROOT, f"output/{v.dataset}/plots/potential_temperature"
)


def plot_theta_850(
    flight: str,
    theta_legs: dict,
    insitu_time: np.ndarray,
    insitu_alt_km: np.ndarray,
    theta_lim: tuple,
    alt_lim: tuple,
) -> str:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, f"{flight.lower()}_theta850.png")

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.tab10.colors
    for i, ((low_start, low_end), leg) in enumerate(theta_legs.items()):
        color = colors[i % len(colors)]
        label = f"leg {low_start}-{low_end} ({leg['theta_850']:.1f} K)"

        # individual descent/ascent measurements
        ax.scatter(leg["leg_times"], leg["leg_thetas"], color=color, s=5, zorder=5)

        # mean line between the two measurements
        if len(leg["leg_times"]) == 2:
            ax.plot(
                leg["leg_times"],
                [leg["theta_850"]] * 2,
                color=color,
                linewidth=2,
                alpha=0.5,
                label=label,
            )
        else:
            ax.scatter([], [], color=color, label=label)  # legend entry only

    ax.set_ylabel("$\\theta_{850}$ (K)")
    ax.set_ylim(theta_lim)

    ax2 = ax.twinx()
    ax2.plot(
        insitu_time,
        insitu_alt_km,
        color="black",
        linewidth=1.0,
        label="Aircraft altitude",
    )
    ax2.set_ylabel("Altitude (km)")
    ax2.set_ylim(alt_lim)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time (UTC)")
    fig.autofmt_xdate()

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)

    ax.set_title(f"{flight}: Potential Temperature at ~850 hPa")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def main():
    all_theta_legs = {}
    all_insitu_time = {}
    all_insitu_alt = {}

    for flight in VERTICAL_LEGS:
        all_theta_legs[flight] = compute_theta_850(flight)
        with open_dataset(v001.dataset, flight) as ds:
            all_insitu_time[flight] = ds[v001.time].values
            all_insitu_alt[flight] = m_to_km(ds[v001.altitude].values)

    all_thetas = [
        t
        for legs in all_theta_legs.values()
        for leg in legs.values()
        for t in leg["leg_thetas"]
    ]
    theta_lim = (min(all_thetas) - 1, max(all_thetas) + 1)

    all_alt = np.concatenate(list(all_insitu_alt.values()))
    alt_lim = (np.nanmin(all_alt), np.nanmax(all_alt))

    for flight, theta_legs in all_theta_legs.items():
        path = plot_theta_850(
            flight,
            theta_legs,
            all_insitu_time[flight],
            all_insitu_alt[flight],
            theta_lim,
            alt_lim,
        )
        for (low_start, low_end), leg in theta_legs.items():
            print(
                f"{flight} leg {low_start}-{low_end}: "
                f"theta_850={leg['theta_850']:.2f} +/- {leg['theta_850_std']:.2f} K, "
                f"h_850={leg['h_850']:.3f} km -> {path}"
            )


if __name__ == "__main__":
    main()
