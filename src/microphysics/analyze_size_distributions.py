"""
Analyze particle size distributions for low-level legs.
"""

import os
import numpy as np

from nc.loader import PROJECT_ROOT
from microphysics.data_loader import build_low_level_dataset, PHASE_ICE
from microphysics.size_distribution import (
    aggregate_size_distribution,
    compute_distribution_statistics,
    bin_by_water_path,
)
from microphysics.plotting import (
    plot_mean_size_distribution,
    plot_size_distribution_scatter,
    plot_binned_size_distributions,
    plot_size_distribution_heatmap,
    plot_integrated_properties,
)

PLOTS_DIR = os.path.join(
    PROJECT_ROOT, "output/microphysics_beta/plots/size_distributions"
)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = build_low_level_dataset(phase_filter=frozenset({PHASE_ICE}))

    # get bin info from first row
    first_row = df.row(0, named=True)
    bin_centers = np.array(first_row["bin_centers"])
    bin_widths = np.array(first_row["bin_widths"])

    # concatenate all concentration arrays
    all_conc = []
    for row in df.iter_rows(named=True):
        all_conc.append(np.array(row["concentration"]))

    concentration = np.column_stack(all_conc)

    sd_mean = aggregate_size_distribution(
        concentration, bin_centers, bin_widths, method="mean"
    )
    stats = compute_distribution_statistics(concentration, bin_centers, bin_widths)

    wvp_binned = bin_by_water_path(df, variable="WVP", n_bins=5, method="quantile")
    lwp_binned = bin_by_water_path(df, variable="LWP", n_bins=5, method="quantile")

    print("Generating plots...")
    plot_mean_size_distribution(sd_mean, os.path.join(PLOTS_DIR, "mean_dNdD_vs_D.png"))

    for metric in ["WVP", "LWP"]:
        plot_size_distribution_scatter(
            df,
            variable=metric,
            output_path=os.path.join(PLOTS_DIR, f"dNdD_vs_D_colored_by_{metric}.png"),
        )
        plot_integrated_properties(
            df,
            output_path=os.path.join(
                PLOTS_DIR, f"integrated_properties_vs_{metric}.png"
            ),
            variable=metric,
        )

    plot_binned_size_distributions(
        wvp_binned,
        variable="WVP",
        output_path=os.path.join(PLOTS_DIR, "dNdD_vs_D_stratified_by_WVP.png"),
    )
    plot_binned_size_distributions(
        lwp_binned,
        variable="LWP",
        output_path=os.path.join(PLOTS_DIR, "dNdD_vs_D_stratified_by_LWP.png"),
    )

    # compute global concentration range for consistent color scale
    conc_min = np.log10(
        np.nanmin(concentration[concentration > 0])
        if np.any(concentration > 0)
        else 1e-10
    )
    conc_max = np.log10(np.nanmax(concentration))

    # generate heatmap for each flight, with one panel per low-level leg
    flights = df["flight"].unique().sort()
    for flight in flights:
        df_flight = df.filter(df["flight"] == flight)
        if df_flight.is_empty():
            continue

        segments = []
        for seg_id in df_flight["segment_id"].unique().sort():
            df_seg = df_flight.filter(df_flight["segment_id"] == seg_id)
            times = []
            conc_list = []
            for row in df_seg.iter_rows(named=True):
                times.append(row["time"])
                conc_list.append(np.array(row["concentration"]))
            segments.append((seg_id, np.column_stack(conc_list), np.array(times)))

        # sort segments by start time
        segments.sort(key=lambda s: s[2][0])

        plot_size_distribution_heatmap(
            segments,
            bin_centers,
            os.path.join(PLOTS_DIR, f"dNdD_heatmap_{flight}.png"),
            vmin=conc_min,
            vmax=conc_max,
            title=f"{flight}: Size distribution on low-level legs",
        )


if __name__ == "__main__":
    main()
