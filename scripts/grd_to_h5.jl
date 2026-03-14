#!/usr/bin/env julia
"""
Convert a GMT earth_relief GRD file to the HDF5 format required by
create_terrain_geometry.jl.

The GRD file is a NetCDF-4 file distributed via the Generic Mapping Tools
(GMT) data server.  This script reads the lon, lat, and z variables and
writes them to an HDF5 file with the structure:

  longitude : (N,)   Float32  – longitude values in degrees, ascending
  latitude  : (M,)   Float32  – latitude values in degrees, ascending
  elevation : (M, N) Float32  – elevation in metres, indexed [lat, lon]

The elevation dataset is written in latitude bands to bound memory use.
Each band loads roughly `band_size` rows (~50 MB for band_size=300 at
15 arc-second resolution).

The elevation dataset is written so that HDF5.jl reads it back as a
(M × N) Julia array with the first index corresponding to latitude, which
is what evaluate_elevation_tiled in create_terrain_geometry.jl expects.

Environment variables
---------------------
CORSIKA_TERRAIN_ELEVATION
    Default output path when --output is not given.
CORSIKA_TERRAIN_DATA
    Fallback directory used when CORSIKA_TERRAIN_ELEVATION is unset;
    output is written to <CORSIKA_TERRAIN_DATA>/earth_relief_15s.h5.

Usage
-----
  julia --project=<project_dir> grd_to_h5.jl INPUT.grd [--output OUTPUT.h5]

Example
-------
  julia --project=. scripts/grd_to_h5.jl earth_relief_15s.grd \\
      --output earth_relief_15s.h5
"""

using ArgParse
using HDF5
using NCDatasets

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

function default_output()::String
    v = get(ENV, "CORSIKA_TERRAIN_ELEVATION", "")
    !isempty(v) && return v
    base = get(ENV, "CORSIKA_TERRAIN_DATA", "")
    !isempty(base) && return joinpath(base, "earth_relief_15s.h5")
    return "earth_relief_15s.h5"
end

function parse_args_main()
    s = ArgParseSettings(
        description = "Convert a GMT earth_relief GRD file to HDF5.",
        add_help    = true,
    )
    @add_arg_table! s begin
        "input"
            help     = "Path to the input GRD file (NetCDF-4)"
            arg_type = String
            required = true
        "--output"
            help     = "Output HDF5 file path [env: CORSIKA_TERRAIN_ELEVATION]"
            arg_type = String
            default  = default_output()
        "--band-size"
            help     = "Number of latitude rows per write band (controls peak memory use)"
            arg_type = Int
            default  = 300
    end
    return ArgParse.parse_args(s)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    args = parse_args_main()
    in_path   = args["input"]
    out_path  = args["output"]
    band_size = args["band-size"]

    println("Input:  $in_path")
    println("Output: $out_path")

    # ── Inspect GRD ─────────────────────────────────────────────────────────
    println("\nInspecting GRD file ...")

    lons, lats, elev_name, lat_first = NCDataset(in_path, "r") do ds
        lon_name  = haskey(ds, "lon") ? "lon"  :
                    haskey(ds, "x")   ? "x"    : "longitude"
        lat_name  = haskey(ds, "lat") ? "lat"  :
                    haskey(ds, "y")   ? "y"    : "latitude"
        elev_name = haskey(ds, "z")   ? "z"    : "elevation"

        lons = Float32.(Array(ds[lon_name]))
        lats = Float32.(Array(ds[lat_name]))

        first_dim = NCDatasets.dimnames(ds[elev_name])[1]
        lat_first = first_dim == lat_name || first_dim == "y"

        lons, lats, elev_name, lat_first
    end

    # Ensure ascending order
    lon_perm = issorted(lons) ? nothing : sortperm(lons)
    lat_perm = issorted(lats) ? nothing : sortperm(lats)
    if lon_perm !== nothing; lons = lons[lon_perm]; end
    if lat_perm !== nothing; lats = lats[lat_perm]; end

    n_lat = length(lats)
    n_lon = length(lons)
    println("  Longitudes: $n_lon  range=[$(lons[1])°, $(lons[end])°]")
    println("  Latitudes:  $n_lat  range=[$(lats[1])°, $(lats[end])°]")
    println("  Elevation first dim is lat: $lat_first")

    # ── Create output file with pre-allocated elevation dataset ─────────────
    out_dir = dirname(abspath(out_path))
    isempty(out_dir) || mkpath(out_dir)

    println("\nWriting $out_path in bands of $band_size rows ...")

    h5open(out_path, "w") do f
        f["longitude"] = lons
        f["latitude"]  = lats
        # Create dataset up front with the right shape so we can write slices.
        # HDF5.jl writes Julia shape (n_lat, n_lon) → h5py shape (n_lon, n_lat),
        # which evaluate_elevation_tiled reads back as Julia (n_lat, n_lon). ✓
        d = create_dataset(f, "elevation", datatype(Float32),
                           dataspace(n_lat, n_lon))

        n_bands = cld(n_lat, band_size)
        NCDataset(in_path, "r") do ds
            for band_idx in 1:n_bands
                lat_lo = (band_idx - 1) * band_size + 1
                lat_hi = min(band_idx * band_size, n_lat)

                band_lats = lat_perm === nothing ? (lat_lo:lat_hi) :
                                                   lat_perm[lat_lo:lat_hi]

                # Read the band from the GRD.
                # NCDatasets applies scale_factor/add_offset, returning
                # Union{Missing,Float64}.  Replace missing with 0 and
                # store as Float32 to match the reference HDF5 file.
                if lat_first
                    raw_f = coalesce.(Array(ds[elev_name][band_lats, :]), 0.0)
                else
                    raw_f = coalesce.(permutedims(Array(ds[elev_name][:, band_lats])), 0.0)
                end

                # Apply longitude permutation if needed
                if lon_perm !== nothing
                    raw_f = raw_f[:, lon_perm]
                end

                d[lat_lo:lat_hi, :] = Float32.(raw_f)
                print("\r  Band $band_idx / $n_bands")
            end
        end
        println()
    end

    sz_mb = round(stat(out_path).size / 1e6; digits=0)
    println("  Wrote $(n_lon) × $(n_lat) grid  ($(sz_mb) MB)")
    println("Done.")
end

main()
