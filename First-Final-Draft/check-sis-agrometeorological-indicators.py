#!/usr/bin/env python3
"""
Inspect a NetCDF file and print a comprehensive summary:
- global attributes
- dimensions
- variables (dtype, dims, shape, attributes)
- simple statistics for numeric variables (min, max, mean) when feasible
- coordinate detection (lat/lon/time) and spatial/time ranges
- projection / grid_mapping info if present

Usage:
    python check-sis-agrometeorological-indicators.py --file /path/to/file.nc
    python check-sis-agrometeorological-indicators.py --file /path/to/file.nc --json out.json --verbose

This script will try to use xarray (recommended). If xarray isn't available, it will fall back to netCDF4.
"""

# ReferenceET-PenmanMonteith-FAO56_C3S-glob-agric_AgERA5_19900101_final-v2.0.0.area-subset.36.25.80.75.32.15.75.5.nc
from __future__ import annotations
import argparse
import json
import os
import sys
import math
from typing import Any, Dict, Optional

DEFAULT_PREFIX = "/Volumes/SSD/data/sis-agrometeorological-indicators/"
DEFAULT_FILENAME = "combined_pet_data_2024.nc"
DEFAULT_PATH = os.path.join(DEFAULT_PREFIX + DEFAULT_FILENAME)
print(DEFAULT_PATH)

# Optional libs
try:
    import xarray as xr  # type: ignore
except Exception:
    xr = None

try:
    from netCDF4 import Dataset  # type: ignore
except Exception:
    Dataset = None

import numpy as np


def human_size(num: float) -> str:
    if num is None or (isinstance(num, float) and math.isnan(num)):
        return "n/a"
    step_unit = 1024.0
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if abs(num) < step_unit:
            return f"{num:3.1f} {x}"
        num /= step_unit
    return f"{num:.1f} PB"


def safe_stats(arr: np.ndarray) -> Dict[str, Any]:
    """Compute stats while avoiding enormous arrays (guard clause)."""
    try:
        size = arr.size
        if size == 0:
            return {"count": 0}
        # Avoid reading very large arrays into memory for stats
        if size > 10_000_000:  # threshold ~10 million elements
            # compute along reduced sampling
            flat = arr.ravel()
            # sample ~1e6 values evenly if too large
            step = max(1, size // 1_000_000)
            sample = flat[::step]
            sample = sample[np.isfinite(sample)]
            if sample.size == 0:
                return {"count": int(size), "sampled": True}
            return {
                "count": int(size),
                "sampled": True,
                "min": float(np.nanmin(sample)),
                "max": float(np.nanmax(sample)),
                "mean": float(np.nanmean(sample)),
            }
        else:
            # safe to compute exact stats
            arrf = arr.flatten()
            arrf = arrf[np.isfinite(arrf)]
            if arrf.size == 0:
                return {"count": int(size)}
            return {
                "count": int(size),
                "min": float(np.nanmin(arrf)),
                "max": float(np.nanmax(arrf)),
                "mean": float(np.nanmean(arrf)),
            }
    except Exception as e:
        return {"error": str(e)}


def inspect_with_xarray(path: str, verbose: bool = False) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    ds = xr.open_dataset(path, decode_times=True, mask_and_scale=True, use_cftime=False, engine=None)
    try:
        info["path"] = os.path.abspath(path)
        info["backend"] = "xarray"
        info["file_size"] = human_size(os.path.getsize(path))
        info["globals"] = {k: v for k, v in ds.attrs.items()}
        info["dimensions"] = {k: int(v) for k, v in ds.sizes.items()}

        # variables summary
        vars_summary: Dict[str, Any] = {}
        for name, var in ds.variables.items():
            # var can be DataArray (with dims) or coordinate
            try:
                dtype = str(var.dtype)
            except Exception:
                dtype = "unknown"
            shape = tuple(var.shape)
            dims = tuple(var.dims)
            attrs = {k: v for k, v in var.attrs.items()}
            stats: Optional[Dict[str, Any]] = None
            # compute stats for numeric non-string types
            try:
                if np.issubdtype(var.dtype, np.number):
                    # load to numpy - beware of memory
                    arr = var.values
                    stats = safe_stats(np.array(arr, dtype=float))
                else:
                    stats = None
            except Exception as e:
                stats = {"error": f"could not compute stats: {e}"}
            vars_summary[name] = {
                "dtype": dtype,
                "dims": dims,
                "shape": shape,
                "attributes": attrs,
                "stats": stats,
            }

        info["variables"] = vars_summary

        # detect coordinate variables like lat/lon/time
        coords = {}
        for coord_name in ds.coords:
            try:
                coord = ds.coords[coord_name]
                coords[coord_name] = {
                    "dtype": str(coord.dtype),
                    "shape": tuple(coord.shape),
                    "attrs": {k: v for k, v in coord.attrs.items()},
                    "min": None,
                    "max": None,
                }
                if np.issubdtype(coord.dtype, np.number):
                    try:
                        cval = coord.values
                        if cval.size > 0:
                            coords[coord_name]["min"] = float(np.nanmin(cval))
                            coords[coord_name]["max"] = float(np.nanmax(cval))
                    except Exception:
                        pass
            except Exception:
                pass
        info["coords"] = coords

        # try to determine lat/lon and time variables
        lat_names = [n for n in ds.coords if "lat" in n.lower()]
        lon_names = [n for n in ds.coords if "lon" in n.lower()]
        time_names = [n for n in ds.coords if "time" in n.lower()]

        info["detected"] = {
            "lat": lat_names,
            "lon": lon_names,
            "time": time_names,
        }

        # bounding boxes for lat/lon if present
        if lat_names and lon_names:
            lat = ds.coords[lat_names[0]].values
            lon = ds.coords[lon_names[0]].values
            try:
                lat_flat = np.array(lat).ravel()
                lon_flat = np.array(lon).ravel()
                info["spatial_bounds"] = {
                    "lat_min": float(np.nanmin(lat_flat)),
                    "lat_max": float(np.nanmax(lat_flat)),
                    "lon_min": float(np.nanmin(lon_flat)),
                    "lon_max": float(np.nanmax(lon_flat)),
                }
            except Exception:
                pass

        # time range
        if time_names:
            try:
                t = ds.coords[time_names[0]]
                tvals = t.values
                # xarray may give numpy datetime64 or cftime objects
                if getattr(tvals, "size", 0) > 0:
                    try:
                        tmin = str(np.nanmin(tvals))
                        tmax = str(np.nanmax(tvals))
                        info["time_range"] = {"start": tmin, "end": tmax}
                    except Exception:
                        info["time_range"] = {"start": None, "end": None}
            except Exception:
                pass

        # grid_mapping or projection info
        proj_info = {}
        # many CF datasets reference a variable name in grid_mapping attribute of variable
        for vname, v in vars_summary.items():
            attrs = v.get("attributes", {})
            gm = attrs.get("grid_mapping") or attrs.get("grid_mapping_name")
            if gm:
                proj_info.setdefault("referenced_by", []).append({vname: gm})
        # also search for typical projection variables
        for k, v in ds.variables.items():
            if "grid_mapping_name" in v.attrs or "spatial_ref" in v.attrs or "crs" in k.lower():
                proj_info[k] = {kk: vv for kk, vv in v.attrs.items()}
        if proj_info:
            info["projection"] = proj_info

        # dataset summary string
        info["summary"] = str(ds)
    finally:
        try:
            ds.close()
        except Exception:
            pass
    return info


def inspect_with_netcdf4(path: str, verbose: bool = False) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    nc = Dataset(path, mode="r")
    try:
        info["path"] = os.path.abspath(path)
        info["backend"] = "netCDF4"
        info["file_size"] = human_size(os.path.getsize(path))
        info["globals"] = {k: getattr(nc, k) for k in nc.ncattrs()}
        info["dimensions"] = {k: len(v) for k, v in nc.dimensions.items()}

        vars_summary: Dict[str, Any] = {}
        for name, var in nc.variables.items():
            dtype = str(var.dtype)
            shape = tuple(var.shape)
            dims = tuple(var.dimensions)
            attrs = {k: getattr(var, k) for k in var.ncattrs()}
            stats = None
            try:
                # read variable (may be large)
                if np.issubdtype(var.dtype, np.number):
                    arr = var[:]
                    stats = safe_stats(np.array(arr, dtype=float))
            except Exception as e:
                stats = {"error": f"could not compute stats: {e}"}
            vars_summary[name] = {
                "dtype": dtype,
                "dims": dims,
                "shape": shape,
                "attributes": attrs,
                "stats": stats,
            }
        info["variables"] = vars_summary

        # coords detection
        coords = {}
        for name, var in nc.variables.items():
            # treat 1D variables with same name as a dimension as coordinate
            if name in nc.dimensions or var.ndim == 1:
                try:
                    arr = var[:]
                    coords[name] = {
                        "dtype": str(var.dtype),
                        "shape": tuple(var.shape),
                        "attrs": {k: getattr(var, k) for k in var.ncattrs()},
                        "min": None,
                        "max": None,
                    }
                    if np.issubdtype(var.dtype, np.number) and arr.size > 0:
                        coords[name]["min"] = float(np.nanmin(arr))
                        coords[name]["max"] = float(np.nanmax(arr))
                except Exception:
                    pass
        info["coords"] = coords

        lat_names = [n for n in coords if "lat" in n.lower()]
        lon_names = [n for n in coords if "lon" in n.lower()]
        time_names = [n for n in coords if "time" in n.lower()]

        info["detected"] = {"lat": lat_names, "lon": lon_names, "time": time_names}

        # spatial bounds
        if lat_names and lon_names:
            try:
                lat = nc.variables[lat_names[0]][:]
                lon = nc.variables[lon_names[0]][:]
                info["spatial_bounds"] = {
                    "lat_min": float(np.nanmin(lat)),
                    "lat_max": float(np.nanmax(lat)),
                    "lon_min": float(np.nanmin(lon)),
                    "lon_max": float(np.nanmax(lon)),
                }
            except Exception:
                pass

        # time range
        if time_names:
            try:
                t = nc.variables[time_names[0]][:]
                if getattr(t, "size", 0) > 0:
                    info["time_range"] = {"start": str(np.nanmin(t)), "end": str(np.nanmax(t))}
            except Exception:
                pass
    finally:
        try:
            nc.close()
        except Exception:
            pass
    return info


def pretty_print(info: Dict[str, Any], verbose: bool = False) -> None:
    sep = "=" * 80
    print(sep)
    print("NetCDF Inspection")
    print(sep)
    print("Path:", info.get("path"))
    print("Backend:", info.get("backend"))
    print("File size:", info.get("file_size"))
    print()

    print("Global attributes:")
    if info.get("globals"):
        for k, v in info["globals"].items():
            print(f"  {k}: {v}")
    else:
        print("  (none)")
    print()

    print("Dimensions:")
    for k, v in info.get("dimensions", {}).items():
        print(f"  {k}: {v}")
    print()

    print("Coords (detected):")
    for cname, cinfo in info.get("coords", {}).items():
        line = f"  {cname}: dtype={cinfo.get('dtype')} shape={cinfo.get('shape')}"
        mn = cinfo.get("min")
        mx = cinfo.get("max")
        if mn is not None or mx is not None:
            line += f" range=({mn}, {mx})"
        print(line)
    print()

    print("Detected lat/lon/time coordinates:", info.get("detected", {}))
    if "spatial_bounds" in info:
        sb = info["spatial_bounds"]
        print("Spatial bounds (lat/lon):", sb)
    if "time_range" in info:
        print("Time range:", info["time_range"])
    print()

    print("Variables:")
    for vname, vinfo in sorted(info.get("variables", {}).items()):
        dtype = vinfo.get("dtype")
        shape = vinfo.get("shape")
        dims = vinfo.get("dims")
        print(f"  {vname}: dtype={dtype} dims={dims} shape={shape}")
        attrs = vinfo.get("attributes") or {}
        if attrs:
            if verbose:
                for ak, av in attrs.items():
                    print(f"      attr: {ak} = {av}")
            else:
                # show a few important attrs
                for key in ("units", "long_name", "standard_name", "grid_mapping"):
                    if key in attrs:
                        print(f"      {key}: {attrs.get(key)}")
        stats = vinfo.get("stats")
        if stats:
            sline = "      stats:"
            if "error" in stats:
                sline += f" {stats['error']}"
            else:
                sline += " " + ", ".join(f"{k}={v}" for k, v in stats.items() if k != "sampled")
                if stats.get("sampled"):
                    sline += " (sampled)"
            print(sline)
    print()

    if "projection" in info:
        print("Projection / grid_mapping info:")
        for k, v in info["projection"].items():
            print(f"  {k}: {v}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Inspect a NetCDF file and print detailed summary.")
    parser.add_argument("--file", "-f", default=DEFAULT_PATH, help="Path to NetCDF file")
    parser.add_argument("--json", "-j", help="Optional path to save output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose attributes for variables")
    args = parser.parse_args()

    path = args.file
    if not os.path.exists(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        print("Provide a valid path with --file", file=sys.stderr)
        sys.exit(2)

    info = None
    if xr is not None:
        try:
            info = inspect_with_xarray(path, verbose=args.verbose)
        except Exception as e:
            print(f"Warning: xarray failed to open file ({e}). Trying netCDF4 fallback.", file=sys.stderr)
            info = None

    if info is None:
        if Dataset is None:
            print("Error: neither xarray nor netCDF4 available to inspect the file.", file=sys.stderr)
            sys.exit(3)
        try:
            info = inspect_with_netcdf4(path, verbose=args.verbose)
        except Exception as e:
            print(f"Error: netCDF4 failed to open file ({e}).", file=sys.stderr)
            sys.exit(4)

    pretty_print(info, verbose=args.verbose)

    if args.json:
        try:
            with open(args.json, "w", encoding="utf8") as fh:
                json.dump(info, fh, indent=2, ensure_ascii=False)
            print(f"Saved JSON summary to {args.json}")
        except Exception as e:
            print(f"Could not save JSON: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
