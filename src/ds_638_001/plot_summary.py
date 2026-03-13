import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from ds_638_001.summary import construct_df
from nc.flights import FLIGHTS
from nc.loader import PROJECT_ROOT, open_dataset
from nc.vars import DS_638_001 as v001

DATASET = "638-001"
PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/638-001/plots/summary")

# configured by feel (reference: https://data.eol.ucar.edu/project/CAESAR)
MIN_LAT = -15
MAX_LAT = 25
MIN_LON = 65
MAX_LON = 80


def plot_altitude(df: pl.DataFrame, ax: plt.Axes, flight_label: str) -> None:
    sns.lineplot(data=df, x="hours_utc", y="alt_km", ax=ax, linewidth=1.0)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda h, _: f"{int(h):02d}:{int((h % 1) * 60):02d}")
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"{flight_label} Altitude Profile")


def setup_map(ax: plt.Axes) -> None:
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=cfeature.COLORS["water"])
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=cfeature.COLORS["land"])
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.25)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, linestyle="--")

    gl = ax.gridlines(draw_labels=True, linewidth=1.0, alpha=1.0)
    gl.top_labels = False
    gl.right_labels = False

    # NB: hardcoded boundaries
    ax.set_extent([MIN_LAT, MAX_LAT, MIN_LON, MAX_LON], crs=ccrs.PlateCarree())
    ax.set_aspect("auto")


def plot_ground_track(df: pl.DataFrame, ax: plt.Axes, label: str) -> None:
    setup_map(ax)
    ax.plot(
        df[v001.longitude].to_numpy(),
        df[v001.latitude].to_numpy(),
        transform=ccrs.PlateCarree(),
        linewidth=1.0,
        color="r",
    )
    ax.set_title(f"{label} Ground Track")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    for flight, date in FLIGHTS.items():
        with open_dataset(DATASET, flight) as ds:
            df = construct_df(ds)

        label = f"CAESAR {flight} ({date})"

        map_proj = ccrs.LambertConformal(
            central_longitude=float(df[v001.longitude].mean()),
            central_latitude=float(df[v001.latitude].mean()),
        )

        fig = plt.figure(figsize=(18, 6))
        ax_alt = fig.add_subplot(1, 2, 1)
        ax_track = fig.add_subplot(1, 2, 2, projection=map_proj)

        plot_altitude(df, ax_alt, label)
        plot_ground_track(df, ax_track, label)

        out_path = os.path.join(PLOTS_DIR, f"{flight.lower()}_summary.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path}")

    # all flights map
    map_proj = ccrs.LambertConformal(
        central_longitude=((MAX_LAT + MIN_LAT) / 2),
        central_latitude=((MAX_LAT + MIN_LAT) / 2),
    )
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=map_proj)

    setup_map(ax)

    for flight, date in FLIGHTS.items():
        with open_dataset(DATASET, flight) as ds:
            df = construct_df(ds)

        ax.plot(
            df[v001.longitude].to_numpy(),
            df[v001.latitude].to_numpy(),
            transform=ccrs.PlateCarree(),
            linewidth=1.25,
            label=f"{flight}",
        )

    ax.legend(loc="lower left", fontsize=11, framealpha=1)
    ax.set_title("Ground track of all research flights")

    out_path = os.path.join(PLOTS_DIR, "all_ground_track.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
