import numpy as np
import xarray as xr

from nc.loader import open_file
from nc.remote import ERA5_SST, SWING3_MODELS
from nc.cache import MEMORY


@MEMORY.cache
def grid_info(ds: xr.Dataset) -> dict:
    lat = ds["lat"].values
    lon = ds["lon"].values
    return {
        "n_lat": len(lat),
        "n_lon": len(lon),
        "lat_res": float(np.diff(lat).mean()),
        "lon_res": float(np.diff(lon).mean()),
        "lat_range": (float(lat.min()), float(lat.max())),
        "lon_range": (float(lon.min()), float(lon.max())),
    }


def main() -> None:
    grids = {}

    for model, path in SWING3_MODELS.items():
        with open_file(path, decode_times=False) as ds:
            grids[model] = grid_info(ds)

    with open_file(ERA5_SST, decode_times=False) as ds:
        grids["ERA5_SST"] = grid_info(ds)

    print(
        f"{'Dataset':<10}  {'n_lat':>5}  {'n_lon':>5}  {'lat_res':>8}  {'lon_res':>8}  {'lat_range':>42}  {'lon_range':>20}"
    )
    print("-" * 110)
    for name, g in grids.items():
        print(
            f"{name:<10}  {g['n_lat']:>5}  {g['n_lon']:>5}"
            f"  {g['lat_res']:>8.4f}  {g['lon_res']:>8.4f}"
            f"  {str(g['lat_range']):>42}  {str(g['lon_range']):>20}"
        )

    # Assert all grids are identical
    reference = grids[next(iter(grids))]
    failures = []
    for name, g in grids.items():
        for key in ("n_lat", "n_lon"):
            if g[key] != reference[key]:
                failures.append(f"{name}: {key} = {g[key]}, expected {reference[key]}")
        for key in ("lat_res", "lon_res"):
            if not np.isclose(g[key], reference[key], atol=1e-4):
                failures.append(
                    f"{name}: {key} = {g[key]:.4f}, expected {reference[key]:.4f}"
                )

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
    else:
        print("\nAll grids match.")


if __name__ == "__main__":
    main()
