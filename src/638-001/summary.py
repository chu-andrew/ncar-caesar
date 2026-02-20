import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import xarray as xr

from nc.loader import DATASET_VARS, PROJECT_ROOT, open_dataset

_vars = DATASET_VARS["638-001"]
LATITUDE = _vars["latitude"]
LONGITUDE = _vars["longitude"]
ZAXIS = _vars["altitude"]
TIME = _vars["time"]

# configured by feel (reference: https://data.eol.ucar.edu/project/CAESAR)
MIN_LAT = -15
MAX_LAT = 25
MIN_LON = 65
MAX_LON = 80


def construct_df(ds: xr.Dataset) -> pl.DataFrame:
    """Convert relevant variables from xarray.Dataset to a Polars DataFrame."""
    if TIME in ds:
        t = ds[TIME]
        ds = ds.assign(hours_utc=t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0)

    if ZAXIS in ds:
        ds = ds.assign(alt_km=ds[ZAXIS] / 1000.0)

    vars_to_keep = {LATITUDE, LONGITUDE, "hours_utc", "alt_km"}
    available_vars = list(vars_to_keep.intersection(ds.variables))

    # convert to df (reset_index flattens coordinates)
    df_pandas = ds[available_vars].to_dataframe().reset_index()
    return pl.from_pandas(df_pandas)


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
        df[LONGITUDE].to_numpy(),
        df[LATITUDE].to_numpy(),
        transform=ccrs.PlateCarree(),
        linewidth=1.0,
        color="r",
    )
    ax.set_title(f"{label} Ground Track")


"""
Summary Plots
"""
DATASET = "638-001"
PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/638-001/plots/summary")

FLIGHTS = {
    "RF01": "2024-02-28",
    "RF02": "2024-02-29",
    "RF03": "2024-03-02",
    "RF04": "2024-03-05",
    "RF05": "2024-03-11",
    "RF06": "2024-03-12",
    "RF07": "2024-03-16",
    "RF09": "2024-04-02",
    "RF10": "2024-04-03",
}


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    for flight, date in FLIGHTS.items():
        with open_dataset(DATASET, flight) as ds:
            df = construct_df(ds)

        label = f"CAESAR {flight} ({date})"

        map_proj = ccrs.LambertConformal(
            central_longitude=float(df[LONGITUDE].mean()),
            central_latitude=float(df[LATITUDE].mean()),
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
            df[LONGITUDE].to_numpy(),
            df[LATITUDE].to_numpy(),
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
