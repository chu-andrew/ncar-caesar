import os

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from netCDF4 import Dataset

from nc.loader import PROJECT_ROOT, list_files, load_dataset

# Derived from globals in the netCDF file
LATITUDE = "LATC"
LONGITUDE = "LONC"
ZAXIS = "GGALT"
TIME = "Time"


def extract_hours(ds: Dataset, time_var: str) -> np.ndarray:
    """Convert time variable (seconds since midnight) to UTC hours."""
    t = ds[time_var][:]
    return t / 3600.0


def plot_altitude(ds: Dataset, ax: plt.Axes, flight_label: str) -> None:
    hours = extract_hours(ds, TIME)
    alt_km = ds[ZAXIS][:] / 1000.0

    sns.lineplot(x=hours, y=alt_km, ax=ax, linewidth=1.0)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda h, _: f"{int(h):02d}:{int((h % 1) * 60):02d}")
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"{flight_label} Altitude Profile")


def plot_ground_track(ds: Dataset, ax: plt.Axes, label: str) -> None:
    lat = ds[LATITUDE][:]
    lon = ds[LONGITUDE][:]

    # background
    ax.add_feature(
        cfeature.OCEAN.with_scale("50m"), facecolor=cartopy.feature.COLORS["water"]
    )
    ax.add_feature(
        cfeature.LAND.with_scale("50m"), facecolor=cartopy.feature.COLORS["land"]
    )
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.25)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, linestyle="--")

    gl = ax.gridlines(draw_labels=True, linewidth=1.0, alpha=1.0)
    gl.top_labels = False
    gl.right_labels = False

    # flight track
    ax.plot(lon, lat, transform=ccrs.PlateCarree(), linewidth=1.0, color="r")

    # NB: hardcoded boundaries
    ax.set_extent([-15, 25, 65, 80], crs=ccrs.PlateCarree())
    ax.set_aspect("auto")

    ax.set_title(f"{label} Ground Track")


"""
Summary Plots
"""
DATASET = "638-001"
PLOTS_DIR = os.path.join(PROJECT_ROOT, "output/638-001/plots/summary")
os.makedirs(PLOTS_DIR, exist_ok=True)

if __name__ == "__main__":
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

    for flight, date in FLIGHTS.items():
        ds = load_dataset(DATASET, flight)
        label = f"CAESAR {flight} ({date})"

        lat_data = ds[LATITUDE][:]
        lon_data = ds[LONGITUDE][:]
        map_proj = ccrs.LambertConformal(
            central_longitude=float((lon_data.min() + lon_data.max()) / 2),
            central_latitude=float((lat_data.min() + lat_data.max()) / 2),
        )

        fig = plt.figure(figsize=(18, 6))
        ax_alt = fig.add_subplot(1, 2, 1)
        ax_track = fig.add_subplot(1, 2, 2, projection=map_proj)

        plot_altitude(ds, ax_alt, label)
        plot_ground_track(ds, ax_track, label)

        out_path = os.path.join(PLOTS_DIR, f"{flight.lower()}_summary.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path}")

        ds.close()
