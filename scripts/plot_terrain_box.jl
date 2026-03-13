#!/usr/bin/env julia
"""
Visualise the terrain mesh and detector box in local ENU coordinates.

Reads the terrain geometry HDF5 file (vertices/faces from
create_terrain_geometry.jl) and the box PLY file (from make_box_scene.jl),
projects everything into a local East-North-Up frame centred on the terrain's
target lat/lon, and writes a three-panel PNG figure:

  Panel 1  Top-down (E–N) view.  Terrain vertices coloured by altitude;
           box footprint outline in red.

  Panel 2  East cross-section (E vs altitude).  Terrain scatter;
           box side outline in red.

  Panel 3  North cross-section (N vs altitude).  Terrain scatter;
           box side outline in red.

Usage
-----
  julia --project=<project_dir> plot_terrain_box.jl \\
      TERRAIN.h5 BOX.ply [options]

  Options:
    --group NAME     HDF5 group for terrain  [default: first group with vertices/faces]
    --output PATH    Output PNG file         [default: terrain_box_check.png]
    --dpi    N       Figure resolution       [default: 150]
"""

using ArgParse
using CairoMakie
using HDF5
using LinearAlgebra
using Statistics

const EARTH_RADIUS_M = 6.371e6

# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

"""
Convert an (N × 3) matrix of ECEF points (metres) into local ENU coordinates
(metres) centred on (lat0_deg, lon0_deg) at Earth's surface.

Returns an (N × 3) matrix with columns [east, north, up].
"""
function ecef_to_enu(ecef::AbstractMatrix, lat0_deg::Float64, lon0_deg::Float64)
    lat0 = deg2rad(lat0_deg)
    lon0 = deg2rad(lon0_deg)

    # Reference ECEF point (on the reference sphere, altitude = 0).
    p0 = EARTH_RADIUS_M .* [cos(lat0)*cos(lon0), cos(lat0)*sin(lon0), sin(lat0)]

    # ENU basis vectors in ECEF.
    east  = [-sin(lon0),  cos(lon0),  0.0]
    north = [-sin(lat0)*cos(lon0), -sin(lat0)*sin(lon0),  cos(lat0)]
    up    = [ cos(lat0)*cos(lon0),  cos(lat0)*sin(lon0),  sin(lat0)]

    R = vcat(east', north', up')   # 3 × 3

    dp = ecef .- p0'               # (N, 3)
    return (R * dp')'              # (N, 3)
end

# ---------------------------------------------------------------------------
# PLY reader (handles the double-precision format written by our scripts)
# ---------------------------------------------------------------------------

function _ply_type_reader(t::AbstractString)
    t in ("float", "float32") && return io -> Float64(ltoh(read(io, Float32)))
    t in ("double", "float64") && return io -> ltoh(read(io, Float64))
    t in ("int", "int32")      && return io -> Int(ltoh(read(io, Int32)))
    t in ("uint", "uint32")    && return io -> Int(ltoh(read(io, UInt32)))
    t in ("uchar", "uint8")    && return io -> Int(read(io, UInt8))
    t in ("short", "int16")    && return io -> Int(ltoh(read(io, Int16)))
    error("Unsupported PLY type: $t")
end

"""Read vertices (N × 3) and face index vectors from a binary_little_endian PLY."""
function read_ply(filepath::AbstractString)
    open(filepath, "r") do io
        # Parse header (ASCII).
        n_verts = 0
        n_faces = 0
        vertex_props = Tuple{String,String}[]
        face_count_type = "uchar"
        face_index_type = "uint"
        current = :none

        while true
            line = strip(readline(io))
            line == "end_header" && break
            tokens = split(line)
            isempty(tokens) && continue

            if tokens[1] == "element"
                current = Symbol(tokens[2])
                tokens[2] == "vertex" && (n_verts = parse(Int, tokens[3]))
                tokens[2] == "face"   && (n_faces = parse(Int, tokens[3]))
            elseif tokens[1] == "property" && current == :vertex && tokens[2] != "list"
                push!(vertex_props, (tokens[end], tokens[2]))
            elseif tokens[1] == "property" && current == :face && tokens[2] == "list"
                face_count_type = tokens[3]
                face_index_type = tokens[4]
            end
        end

        xi = findfirst(p -> p[1] == "x", vertex_props)
        yi = findfirst(p -> p[1] == "y", vertex_props)
        zi = findfirst(p -> p[1] == "z", vertex_props)

        readers = [_ply_type_reader(p[2]) for p in vertex_props]

        verts = zeros(Float64, n_verts, 3)
        for i in 1:n_verts
            vals = [r(io) for r in readers]
            verts[i, 1] = vals[xi]
            verts[i, 2] = vals[yi]
            verts[i, 3] = vals[zi]
        end

        cnt_reader = _ply_type_reader(face_count_type)
        idx_reader = _ply_type_reader(face_index_type)
        faces = Vector{Vector{Int}}(undef, n_faces)
        for j in 1:n_faces
            cnt = cnt_reader(io)
            faces[j] = [idx_reader(io) for _ in 1:cnt]
        end

        return verts, faces
    end
end

# ---------------------------------------------------------------------------
# Box outline extraction
# ---------------------------------------------------------------------------

"""
Build the 12 outline edges of the box hull from its PLY vertices in ENU space.

Finds approximate axis-aligned bounds and returns six quads (as 5-point closed
polygons) representing the six faces, plus projects to the three view planes.
"""
function box_outlines_enu(enu::AbstractMatrix)
    # Min/max in each ENU axis.
    e_lo, e_hi = extrema(enu[:, 1])
    n_lo, n_hi = extrema(enu[:, 2])
    u_lo, u_hi = extrema(enu[:, 3])

    # Top-down footprint (E–N plane): rectangle at mean up.
    en_rect = [e_lo n_lo; e_hi n_lo; e_hi n_hi; e_lo n_hi; e_lo n_lo]

    # East cross-section (E–altitude): rectangle.
    eu_rect = [e_lo u_lo; e_hi u_lo; e_hi u_hi; e_lo u_hi; e_lo u_lo]

    # North cross-section (N–altitude): rectangle.
    nu_rect = [n_lo u_lo; n_hi u_lo; n_hi u_hi; n_lo u_hi; n_lo u_lo]

    return en_rect, eu_rect, nu_rect
end

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

function parse_args_main()
    s = ArgParseSettings(
        description = "Plot terrain + box geometry in local ENU coordinates.",
        add_help    = true,
    )
    @add_arg_table! s begin
        "terrain_h5"
            help     = "Terrain geometry HDF5 file (from create_terrain_geometry.jl)"
            arg_type = String
            required = true
        "box_ply"
            help     = "Box PLY file (from make_box_scene.jl)"
            arg_type = String
            required = true
        "--group"
            help     = "HDF5 group for terrain vertices/faces"
            arg_type = String
            default  = ""
        "--output"
            help     = "Output PNG file path"
            arg_type = String
            default  = "terrain_box_check.png"
        "--dpi"
            help     = "Figure resolution (dots per inch)"
            arg_type = Int
            default  = 150
    end
    return ArgParse.parse_args(s)
end

function find_default_group(fid::HDF5.File)::String
    for name in keys(fid)
        g = fid[name]
        if isa(g, HDF5.Group) && haskey(g, "vertices") && haskey(g, "faces")
            return name
        end
    end
    error("No group with 'vertices' and 'faces' found in HDF5 file.")
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    args = parse_args_main()

    # ── Load terrain ──────────────────────────────────────────────────────────
    println("Reading terrain: $(args["terrain_h5"])")
    terrain_verts_ecef, lat0_deg, lon0_deg = h5open(args["terrain_h5"], "r") do f
        gname = isempty(args["group"]) ? find_default_group(f) : args["group"]
        println("  Group: $gname")
        g = f[gname]
        # HDF5.jl reads HDF5 (3, N) as Julia (N, 3) — no transpose needed.
        v = Float64.(read(g["vertices"]))
        lat0 = read(HDF5.attributes(g)["lat_deg"])
        lon0 = read(HDF5.attributes(g)["lon_deg"])
        v, Float64(lat0), Float64(lon0)
    end
    n_terrain = size(terrain_verts_ecef, 1)
    println("  $(n_terrain) terrain vertices  (lat=$(round(lat0_deg,digits=4)) deg, " *
            "lon=$(round(lon0_deg,digits=4)) deg)")

    # ── Load box ──────────────────────────────────────────────────────────────
    println("Reading box: $(args["box_ply"])")
    box_verts_ecef, _ = read_ply(args["box_ply"])
    n_box = size(box_verts_ecef, 1)
    println("  $(n_box) box vertices")

    # ── Project to local ENU ──────────────────────────────────────────────────
    terrain_enu = ecef_to_enu(terrain_verts_ecef, lat0_deg, lon0_deg)
    box_enu     = ecef_to_enu(box_verts_ecef,     lat0_deg, lon0_deg)

    terrain_alt = terrain_enu[:, 3]   # "up" component ≈ altitude above ref sphere

    println("  Terrain ENU extent (km):")
    println("    East:  $(round(minimum(terrain_enu[:,1])/1e3, digits=1)) .. " *
            "$(round(maximum(terrain_enu[:,1])/1e3, digits=1))")
    println("    North: $(round(minimum(terrain_enu[:,2])/1e3, digits=1)) .. " *
            "$(round(maximum(terrain_enu[:,2])/1e3, digits=1))")
    println("  Box ENU position (m):")
    println("    East:  $(round(minimum(box_enu[:,1]),digits=1)) .. " *
            "$(round(maximum(box_enu[:,1]),digits=1))")
    println("    North: $(round(minimum(box_enu[:,2]),digits=1)) .. " *
            "$(round(maximum(box_enu[:,2]),digits=1))")
    println("    Up:    $(round(minimum(box_enu[:,3]),digits=1)) .. " *
            "$(round(maximum(box_enu[:,3]),digits=1))")

    # Outline rectangles for the box in each projection.
    en_rect, eu_rect, nu_rect = box_outlines_enu(box_enu)

    # ── Figure ────────────────────────────────────────────────────────────────
    println("Building figure ...")
    set_theme!(theme_latexfonts())

    fig = Figure(size = (1400, 480))

    # Shared colour scale: altitude in metres.
    alt_lo, alt_hi = extrema(terrain_alt)

    cmap  = :terrain
    clims = (alt_lo, alt_hi)

    # Scale factors for axis labels.
    km = 1e3

    # ── Panel 1: top-down (E–N) ───────────────────────────────────────────────
    ax1 = Axis(fig[1, 1],
               xlabel = "East  (km)",
               ylabel = "North  (km)",
               title  = "Top-down  (E–N)",
               aspect = DataAspect())

    sc1 = scatter!(ax1,
                   terrain_enu[:, 1] ./ km, terrain_enu[:, 2] ./ km,
                   color      = terrain_alt,
                   colormap   = cmap,
                   colorrange = clims,
                   markersize = 3,
                   strokewidth = 0)

    # Box footprint outline (convert to km for this axis).
    lines!(ax1, en_rect[:, 1] ./ km, en_rect[:, 2] ./ km,
           color = :red, linewidth = 2, label = "box")

    # Mark centre of box.
    box_centre = mean(box_enu, dims=1)[:]
    scatter!(ax1, [box_centre[1] / km], [box_centre[2] / km],
             color = :red, markersize = 8, marker = :cross)

    # ── Panel 2: east cross-section (E–altitude) ──────────────────────────────
    ax2 = Axis(fig[1, 2],
               xlabel = "East  (km)",
               ylabel = "Altitude  (m)",
               title  = "East cross-section")

    scatter!(ax2,
             terrain_enu[:, 1] ./ km, terrain_alt,
             color      = terrain_alt,
             colormap   = cmap,
             colorrange = clims,
             markersize = 3,
             strokewidth = 0)

    lines!(ax2, eu_rect[:, 1] ./ km, eu_rect[:, 2],
           color = :red, linewidth = 2)

    # ── Panel 3: north cross-section (N–altitude) ─────────────────────────────
    ax3 = Axis(fig[1, 3],
               xlabel = "North  (km)",
               ylabel = "Altitude  (m)",
               title  = "North cross-section")

    scatter!(ax3,
             terrain_enu[:, 2] ./ km, terrain_alt,
             color      = terrain_alt,
             colormap   = cmap,
             colorrange = clims,
             markersize = 3,
             strokewidth = 0)

    lines!(ax3, nu_rect[:, 1] ./ km, nu_rect[:, 2],
           color = :red, linewidth = 2)

    # Shared colour bar.
    Colorbar(fig[1, 4], sc1,
             label  = "Altitude  (m)",
             height = Relative(0.85))

    Label(fig[0, 1:3],
          "Terrain + detector box  —  lat=$(round(lat0_deg,digits=3))°  " *
          "lon=$(round(lon0_deg,digits=3))°",
          fontsize = 14, tellwidth = false)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = dirname(abspath(args["output"]))
    isempty(out_dir) || mkpath(out_dir)

    save(args["output"], fig, px_per_unit = args["dpi"] / 72)
    println("Saved: $(args["output"])")
    println("Done.")
end

main()
