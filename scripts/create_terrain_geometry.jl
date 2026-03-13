#!/usr/bin/env julia
"""
Create a terrain geometry HDF5 file from a full closed-sphere triangulation.

Loads a base sphere triangulation from triangulation.h5, rotates it so the
densest sampling region is centred on the target location, evaluates terrain
elevation at each vertex using earth_relief_15s.h5, and writes the resulting
3-D ECEF mesh to an HDF5 file.

All triangles from the triangulation are retained, producing a closed
(watertight) spherical mesh suitable for use as a volume in CORSIKA 8.
Elevation is evaluated globally in 10°×10° tiles to bound memory use.

The base triangulation is densest near the north pole.  Rotating the north
pole to the target location therefore produces the finest mesh near the
target.

Output HDF5 group contains:
  vertices : (3, N) Float64  -- ECEF coordinates in metres (x, y, z)
  faces    : (3, M) Int64    -- triangle vertex indices, 0-based

Group attributes:
  lat_deg, lon_deg, triangulation_group, face_indexing

Environment variables
---------------------
CORSIKA_TERRAIN_TRIANGULATION
    Path to triangulation.h5.
    Fallback: \$CORSIKA_TERRAIN_DATA/triangulation.h5

CORSIKA_TERRAIN_ELEVATION
    Path to earth_relief_15s.h5.
    Fallback: \$CORSIKA_TERRAIN_DATA/earth_relief_15s.h5

CORSIKA_TERRAIN_DATA
    Base directory used when the per-file variables above are unset.

Usage
-----
  julia --project=<project_dir> create_terrain_geometry.jl LAT LON [options]

  Options:
    --triangulation PATH   Path to triangulation.h5
    --elevation PATH       Path to earth_relief_15s.h5
    --group NAME           Triangulation group name  [default: base_triangulation_30000]
    --output PATH          Output HDF5 file          [default: terrain.h5]
    --output-group NAME    HDF5 group name           [default: terrain]
"""

using ArgParse
using Dierckx
using HDF5
using LinearAlgebra
using Rotations

const EARTH_RADIUS_M = 6.371e6

# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

"""Convert (lon, lat) vectors in radians to an (N × 3) matrix of unit vectors."""
function longlat_to_cart(lon_rad::AbstractVector, lat_rad::AbstractVector)
    cos_lat = cos.(lat_rad)
    hcat(cos_lat .* cos.(lon_rad), cos_lat .* sin.(lon_rad), sin.(lat_rad))
end

"""
Convert an (N × 3) matrix of (approximately) unit vectors to
(lon_rad, lat_rad) as two length-N vectors.
"""
function cart_to_longlat(xyz::AbstractMatrix)
    lon = atan.(xyz[:, 2], xyz[:, 1])
    lat = asin.(clamp.(xyz[:, 3], -1.0, 1.0))
    lon, lat
end

"""
Return the rotation that maps the north pole [0, 0, 1] to the point at
(lon0_rad, lat0_rad).  Applying this to the base triangulation places the
densest sampling region at the target location.
"""
function build_rotation_north_to_target(lon0_rad::Float64, lat0_rad::Float64)
    target = longlat_to_cart([lon0_rad], [lat0_rad])[1, :]
    north  = [0.0, 0.0, 1.0]

    axis_vec = cross(north, target)
    axis_norm = norm(axis_vec)

    if axis_norm < 1e-10
        return one(RotMatrix{3, Float64})
    end

    axis  = axis_vec / axis_norm
    angle = acos(clamp(dot(north, target), -1.0, 1.0))
    return AngleAxis(angle, axis[1], axis[2], axis[3])
end

# ---------------------------------------------------------------------------
# Elevation evaluation
# ---------------------------------------------------------------------------

"""
Return terrain elevation (metres) at each (lon_deg, lat_deg) query point.

Loads only the bounding-box tile from earth_relief_15s.h5, fits a degree-2
2-D spline, and evaluates it at the query coordinates.  Results are clamped
to a minimum of 1 m.

HDF5.jl reads the (86400 × 43200) h5py dataset as a (43200 × 86400) Julia
array, so the first Julia index corresponds to latitude and the second to
longitude.
"""
function evaluate_elevation(
    lon_deg::AbstractVector,
    lat_deg::AbstractVector,
    elevation_path::AbstractString;
    pad_deg::Float64 = 2.0,
)
    lon_min = minimum(lon_deg) - pad_deg
    lon_max = maximum(lon_deg) + pad_deg
    lat_min = minimum(lat_deg) - pad_deg
    lat_max = maximum(lat_deg) + pad_deg

    h5open(elevation_path, "r") do f
        lons = Float64.(read(f["longitude"]))
        lats = Float64.(read(f["latitude"]))

        lon_min = max(lon_min, lons[1])
        lon_max = min(lon_max, lons[end])
        lat_min = max(lat_min, lats[1])
        lat_max = min(lat_max, lats[end])

        i0 = searchsortedfirst(lons, lon_min)
        i1 = min(searchsortedlast(lons, lon_max) + 1, size(f["elevation"], 2))
        j0 = searchsortedfirst(lats, lat_min)
        j1 = min(searchsortedlast(lats, lat_max) + 1, size(f["elevation"], 1))

        sub_lons = lons[i0:i1]
        sub_lats = lats[j0:j1]

        # HDF5.jl: elevation[lat_idx, lon_idx] → shape (n_lat_sub, n_lon_sub)
        sub_elev = Float64.(f["elevation"][j0:j1, i0:i1])

        # Dierckx.Spline2D(x, y, z) expects z[i,j] = f(x[i], y[j])
        # with x = lons, y = lats → z must be (n_lon, n_lat) = sub_elev'
        itp = Spline2D(sub_lons, sub_lats, sub_elev'; kx=2, ky=2, s=0.0)

        return [max(itp(lo, la), 1.0) for (lo, la) in zip(lon_deg, lat_deg)]
    end
end

"""
Evaluate terrain elevation globally by processing vertices in 10°×10° tiles.

This avoids loading the full DEM (which would be ~28 GB) into memory at once.
Each tile loads roughly 46 MB.  Longitudes are normalised to [-180, 180]
before tiling.
"""
function evaluate_elevation_tiled(
    lon_deg::AbstractVector,
    lat_deg::AbstractVector,
    elevation_path::AbstractString;
    tile_deg::Float64 = 10.0,
    pad_deg::Float64  = 1.0,
)
    n = length(lon_deg)
    elevations = fill(1.0, n)

    # Normalise longitudes to [-180, 180].
    lon_norm = mod.(lon_deg .+ 180.0, 360.0) .- 180.0

    lon_edges = -180.0:tile_deg:180.0
    lat_edges = -90.0:tile_deg:90.0

    n_tiles = 0
    for la_lo in lat_edges[1:end-1], lo_lo in lon_edges[1:end-1]
        la_hi = la_lo + tile_deg
        lo_hi = lo_lo + tile_deg
        mask = findall((lon_norm .>= lo_lo) .& (lon_norm .< lo_hi) .&
                       (lat_deg  .>= la_lo) .& (lat_deg  .< la_hi))
        isempty(mask) && continue
        elevations[mask] = evaluate_elevation(
            lon_norm[mask], lat_deg[mask], elevation_path; pad_deg = pad_deg)
        n_tiles += 1
    end
    println("  Evaluated elevation over $(n_tiles) tiles")
    return elevations
end

# ---------------------------------------------------------------------------
# Default path helpers
# ---------------------------------------------------------------------------

function default_path(env_var::String, filename::String)::String
    v = get(ENV, env_var, "")
    !isempty(v) && return v
    base = get(ENV, "CORSIKA_TERRAIN_DATA", "")
    !isempty(base) && return joinpath(base, filename)
    return joinpath(@__DIR__, "..", "resources", filename)
end

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

function parse_args_main()
    s = ArgParseSettings(
        description = "Create a terrain geometry HDF5 file for a given lat/lon.",
        add_help    = true,
    )
    @add_arg_table! s begin
        "lat"
            help     = "Target latitude  (degrees, -90..90)"
            arg_type = Float64
            required = true
        "lon"
            help     = "Target longitude (degrees, -180..180)"
            arg_type = Float64
            required = true
        "--triangulation"
            help     = "Path to triangulation.h5 [env: CORSIKA_TERRAIN_TRIANGULATION]"
            arg_type = String
            default  = default_path("CORSIKA_TERRAIN_TRIANGULATION", "triangulation.h5")
        "--elevation"
            help     = "Path to earth_relief_15s.h5 [env: CORSIKA_TERRAIN_ELEVATION]"
            arg_type = String
            default  = default_path("CORSIKA_TERRAIN_ELEVATION", "earth_relief_15s.h5")
        "--group"
            help     = "Triangulation group name in triangulation.h5"
            arg_type = String
            default  = "base_triangulation_30000"
        "--output"
            help     = "Output HDF5 file path"
            arg_type = String
            default  = "terrain.h5"
        "--output-group"
            help     = "HDF5 group name in the output file"
            arg_type = String
            default  = "terrain"
    end
    return ArgParse.parse_args(s)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    args = parse_args_main()

    lat0_deg = args["lat"]
    lon0_deg = args["lon"]
    lat0_rad = deg2rad(lat0_deg)
    lon0_rad = deg2rad(lon0_deg)

    println("Target:         lat=$(lat0_deg) deg,  lon=$(lon0_deg) deg")
    println("Triangulation:  $(args["triangulation"])  (group: $(args["group"]))")
    println("Elevation data: $(args["elevation"])")

    # ── Load base triangulation ─────────────────────────────────────────────
    println("\nLoading triangulation ...")
    lon_base_deg, lat_base_deg, raw_faces = h5open(args["triangulation"], "r") do f
        grp = f[args["group"]]
        # HDF5.jl reverses dimensions vs h5py: (2, N) in h5py → (N, 2) in Julia.
        raw_verts = Float64.(read(grp["vertices"]))  # (N, 2) degrees: col1=lon, col2=lat
        # (3, M) in h5py → (M, 3) in Julia.
        raw_faces = Int64.(read(grp["faces"]))        # (M, 3), 1-based
        raw_verts[:, 1], raw_verts[:, 2], raw_faces
    end

    n_verts_raw = length(lon_base_deg)
    n_faces_raw = size(raw_faces, 1)
    println("  Loaded $(n_verts_raw) vertices, $(n_faces_raw) faces (1-based)")

    lon_base_rad = deg2rad.(lon_base_deg)
    lat_base_rad = deg2rad.(lat_base_deg)

    # Convert to unit 3-D vectors; north pole = [0, 0, 1] = highest density.
    xyz = longlat_to_cart(lon_base_rad, lat_base_rad)  # (N, 3)

    # ── Rotate: north pole → target ─────────────────────────────────────────
    println("Rotating triangulation to target location ...")
    rot = build_rotation_north_to_target(lon0_rad, lat0_rad)
    # rot * v for each row; easier to apply column-wise then transpose back.
    xyz_rot = Matrix((rot * xyz')')   # (N, 3)

    # ── Use full triangulation (closed sphere) ───────────────────────────────
    println("  Using all $(n_verts_raw) vertices, $(n_faces_raw) faces")
    xyz_rot_filtered = xyz_rot
    faces_filtered   = raw_faces .- 1   # convert 1-based → 0-based, (M, 3)
    n_kept           = n_verts_raw
    n_faces_kept     = n_faces_raw

    # ── Rotated lon/lat for elevation lookup ─────────────────────────────────
    lon_rot_rad, lat_rot_rad = cart_to_longlat(xyz_rot_filtered)
    lon_rot_deg  = rad2deg.(lon_rot_rad)
    lat_rot_deg  = rad2deg.(lat_rot_rad)
    lon_for_elev = mod.(lon_rot_deg .+ 180.0, 360.0) .- 180.0

    # ── Evaluate elevation ───────────────────────────────────────────────────
    println("Evaluating terrain elevation ...")
    elevations = evaluate_elevation_tiled(lon_for_elev, lat_rot_deg, args["elevation"])
    println("  Elevation range: $(round(Int, minimum(elevations))) .. " *
            "$(round(Int, maximum(elevations))) m  " *
            "(mean: $(round(Int, sum(elevations)/length(elevations))) m)")

    # ── Compute 3-D ECEF vertices ────────────────────────────────────────────
    println("Computing 3-D ECEF vertices ...")
    # xyz_rot_filtered rows are unit vectors; scale by Earth radius + elevation.
    norms = [norm(xyz_rot_filtered[i, :]) for i in 1:n_kept]
    r = EARTH_RADIUS_M .+ elevations
    vertices_3d = xyz_rot_filtered .* reshape(r ./ norms, :, 1)  # (n_kept, 3)

    radii = [norm(vertices_3d[i, :]) for i in 1:n_kept]
    println("  Geocentric radius: $(round(Int, minimum(radii))) .. " *
            "$(round(Int, maximum(radii))) m")

    # ── Save output ──────────────────────────────────────────────────────────
    out_dir = dirname(abspath(args["output"]))
    isempty(out_dir) || mkpath(out_dir)

    println("\nWriting $(args["output"])  (group: $(args["output-group"])) ...")
    h5open(args["output"], "w") do f
        g = create_group(f, args["output-group"])
        # Write (N, 3) Julia → HDF5.jl stores as (3, N) on disk (h5py convention).
        g["vertices"] = Float64.(vertices_3d)    # Julia (N, 3) → HDF5 (3, N)
        g["faces"]    = Int64.(faces_filtered)   # Julia (M, 3) → HDF5 (3, M), 0-based
        attributes(g)["lat_deg"]             = lat0_deg
        attributes(g)["lon_deg"]             = lon0_deg
        attributes(g)["triangulation_group"] = args["group"]
        attributes(g)["face_indexing"]       = "0-based"
    end

    println("  Wrote $(n_kept) vertices, $(n_faces_kept) faces")
    println("Done.")
end

main()
