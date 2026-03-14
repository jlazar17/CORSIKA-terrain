#!/usr/bin/env julia
"""
Plot shower particle hits on the detector box and terrain mesh.

Two-panel figure in local ENU coordinates:

  Left   – 300 m top-down view: terrain triangles coloured by altitude,
            box footprint (red), particle hits coloured by log10(KE/GeV).

  Right  – Zoom ±15 m around box centre: box triangle edges (red),
            particle hits coloured by log10(KE/GeV).

Usage
-----
  julia --project=. scripts/plot_shower_hits.jl \\
      TERRAIN.h5 BOX.ply PARTICLES.parquet [options]

  Options:
    --output PATH    Output PNG file   [default: shower_hits.png]
    --radius M       Terrain radius in left panel (m)  [default: 300]
    --zoom M         Half-width of zoom panel (m)      [default: 15]
    --dpi N          Resolution                        [default: 200]
"""

using ArgParse
using CairoMakie
using HDF5
using LinearAlgebra
using Parquet2
using Statistics

const EARTH_RADIUS_M = 6.371e6

# ---------------------------------------------------------------------------
# Shared helpers (same as plot_terrain_box.jl)
# ---------------------------------------------------------------------------

function ecef_to_enu(ecef::AbstractMatrix, lat0_deg::Float64, lon0_deg::Float64)
    lat0 = deg2rad(lat0_deg)
    lon0 = deg2rad(lon0_deg)
    p0   = EARTH_RADIUS_M .* [cos(lat0)*cos(lon0), cos(lat0)*sin(lon0), sin(lat0)]
    east  = [-sin(lon0),  cos(lon0),  0.0]
    north = [-sin(lat0)*cos(lon0), -sin(lat0)*sin(lon0),  cos(lat0)]
    up    = [ cos(lat0)*cos(lon0),  cos(lat0)*sin(lon0),  sin(lat0)]
    R  = vcat(east', north', up')
    dp = ecef .- p0'
    return (R * dp')'
end

function _ply_type_reader(t::AbstractString)
    t in ("float",  "float32") && return io -> Float64(ltoh(read(io, Float32)))
    t in ("double", "float64") && return io -> ltoh(read(io, Float64))
    t in ("int",    "int32")   && return io -> Int(ltoh(read(io, Int32)))
    t in ("uint",   "uint32")  && return io -> Int(ltoh(read(io, UInt32)))
    t in ("uchar",  "uint8")   && return io -> Int(read(io, UInt8))
    t in ("short",  "int16")   && return io -> Int(ltoh(read(io, Int16)))
    error("Unsupported PLY type: $t")
end

function read_ply(filepath::AbstractString)
    open(filepath, "r") do io
        n_verts = 0; n_faces = 0
        vertex_props = Tuple{String,String}[]
        face_count_type = "uchar"; face_index_type = "uint"
        current = :none
        while true
            line = strip(readline(io))
            line == "end_header" && break
            tokens = split(line); isempty(tokens) && continue
            if tokens[1] == "element"
                current = Symbol(tokens[2])
                tokens[2] == "vertex" && (n_verts = parse(Int, tokens[3]))
                tokens[2] == "face"   && (n_faces = parse(Int, tokens[3]))
            elseif tokens[1] == "property" && current == :vertex && tokens[2] != "list"
                push!(vertex_props, (tokens[end], tokens[2]))
            elseif tokens[1] == "property" && current == :face && tokens[2] == "list"
                face_count_type = tokens[3]; face_index_type = tokens[4]
            end
        end
        xi = findfirst(p -> p[1] == "x", vertex_props)
        yi = findfirst(p -> p[1] == "y", vertex_props)
        zi = findfirst(p -> p[1] == "z", vertex_props)
        readers = [_ply_type_reader(p[2]) for p in vertex_props]
        verts = zeros(Float64, n_verts, 3)
        for i in 1:n_verts
            vals = [r(io) for r in readers]
            verts[i, 1] = vals[xi]; verts[i, 2] = vals[yi]; verts[i, 3] = vals[zi]
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
# Parquet reader
# ---------------------------------------------------------------------------

"""
Read particle positions and kinetic energies from a CORSIKA 8 parquet file.

Returns (xyz_ecef, ke_gev, pdg) where xyz_ecef is (N×3) Float64 in metres,
ke_gev is (N,) Float64 in GeV, and pdg is (N,) Int64.
Prints available columns so the user can check.
"""
function read_particles(path::AbstractString)
    ds  = Parquet2.Dataset(path)
    tbl = Parquet2.read(ds)
    cols = propertynames(tbl)
    println("  Parquet columns: ", join(string.(cols), ", "))

    # Position: try "x","y","z" then "position_x" etc.
    function getcol(names...)
        for n in names
            sym = Symbol(n)
            sym in cols && return Vector{Float64}(getproperty(tbl, sym))
        end
        error("None of $(names) found in parquet columns: $(cols)")
    end

    x  = getcol("x", "position_x", "X")
    y  = getcol("y", "position_y", "Y")
    z  = getcol("z", "position_z", "Z")

    # Kinetic energy: try eV then GeV columns
    ke_raw = getcol("kinetic_energy", "energy", "ekin", "KineticEnergy")
    # Determine unit: if median > 1e6 it is likely in eV
    ke_gev = median(ke_raw) > 1e6 ? ke_raw ./ 1e9 : ke_raw

    # PDG code (optional)
    pdg_sym = findfirst(n -> n in cols, [Symbol("pdg_code"), Symbol("pdg"),
                                          Symbol("particle"), Symbol("code")])
    pdg = pdg_sym !== nothing ? Vector{Int}(getproperty(tbl, cols[pdg_sym])) :
                                fill(0, length(x))

    return hcat(x, y, z), ke_gev, pdg
end

# ---------------------------------------------------------------------------
# Terrain helpers
# ---------------------------------------------------------------------------

"""Filter terrain faces whose centroid ENU distance (E-N only) from origin < radius."""
function filter_terrain_faces(verts_enu::AbstractMatrix,
                               faces::AbstractMatrix,
                               radius::Float64)
    keep = Int[]
    for i in 1:size(faces, 1)
        idx = faces[i, :] .+ 1   # 0-based → 1-based
        ce  = mean(verts_enu[idx, 1])
        cn  = mean(verts_enu[idx, 2])
        sqrt(ce^2 + cn^2) <= radius && push!(keep, i)
    end
    return keep
end

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

function parse_args_main()
    s = ArgParseSettings(
        description = "Plot shower particle hits on terrain + box mesh.",
        add_help    = true,
    )
    @add_arg_table! s begin
        "terrain_h5"
            help = "Terrain geometry HDF5 (from create_terrain_geometry.jl)"
            arg_type = String; required = true
        "box_ply"
            help = "Box PLY file (from make_box_scene.jl)"
            arg_type = String; required = true
        "particles_parquet"
            help = "Particles parquet (from terrain_shower output/particles/)"
            arg_type = String; required = true
        "--output"
            help = "Output PNG file path"
            arg_type = String; default = "shower_hits.png"
        "--radius"
            help = "Terrain display radius in left panel (m)"
            arg_type = Float64; default = 300.0
        "--zoom"
            help = "Half-width of zoom panel around box (m)"
            arg_type = Float64; default = 15.0
        "--dpi"
            help = "Resolution (dots per inch)"
            arg_type = Int; default = 200
        "--group"
            help = "HDF5 group for terrain vertices/faces"
            arg_type = String; default = ""
    end
    return ArgParse.parse_args(s)
end

function find_default_group(fid::HDF5.File)::String
    for name in keys(fid)
        g = fid[name]
        isa(g, HDF5.Group) && haskey(g, "vertices") && haskey(g, "faces") && return name
    end
    error("No group with 'vertices' and 'faces' found.")
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    args = parse_args_main()

    # ── Load terrain ──────────────────────────────────────────────────────────
    println("Reading terrain: $(args["terrain_h5"])")
    terrain_verts_ecef, terrain_faces_raw, lat0_deg, lon0_deg =
        h5open(args["terrain_h5"], "r") do f
            gname = isempty(args["group"]) ? find_default_group(f) : args["group"]
            println("  Group: $gname")
            g = f[gname]
            v  = Float64.(read(g["vertices"]))   # (N, 3)
            fa = Int.(read(g["faces"]))           # (M, 3), 0-based
            lat0 = Float64(read(HDF5.attributes(g)["lat_deg"]))
            lon0 = Float64(read(HDF5.attributes(g)["lon_deg"]))
            v, fa, lat0, lon0
        end
    println("  $(size(terrain_verts_ecef,1)) vertices, $(size(terrain_faces_raw,1)) faces")

    # ── Load box ──────────────────────────────────────────────────────────────
    println("Reading box: $(args["box_ply"])")
    box_verts_ecef, box_faces_raw = read_ply(args["box_ply"])
    println("  $(size(box_verts_ecef,1)) vertices, $(length(box_faces_raw)) faces")

    # ── Load particles ────────────────────────────────────────────────────────
    println("Reading particles: $(args["particles_parquet"])")
    part_xyz_ecef, part_ke_gev, part_pdg = read_particles(args["particles_parquet"])
    n_part = size(part_xyz_ecef, 1)
    println("  $(n_part) particle hits")
    if n_part == 0
        println("  No particles found — plot will show geometry only.")
    end

    # ── Project to ENU ────────────────────────────────────────────────────────
    terrain_enu  = ecef_to_enu(terrain_verts_ecef, lat0_deg, lon0_deg)
    box_enu      = ecef_to_enu(box_verts_ecef,     lat0_deg, lon0_deg)
    part_enu     = n_part > 0 ?
                   ecef_to_enu(part_xyz_ecef, lat0_deg, lon0_deg) :
                   zeros(0, 3)

    # ── Filter terrain faces within radius ────────────────────────────────────
    nearby = filter_terrain_faces(terrain_enu, terrain_faces_raw, args["radius"])
    println("  $(length(nearby)) terrain faces within $(args["radius"]) m")

    # ── Colour scale for kinetic energy ──────────────────────────────────────
    log_ke = n_part > 0 ? log10.(clamp.(part_ke_gev, 1e-4, Inf)) : Float64[]
    ke_lo  = n_part > 0 ? minimum(log_ke) : -2.0
    ke_hi  = n_part > 0 ? maximum(log_ke) :  4.0

    # Terrain altitude colour limits from visible faces.
    vis_verts = unique(vcat([terrain_faces_raw[i, :] .+ 1 for i in nearby]...))
    alt_lo = isempty(vis_verts) ? 0.0 : minimum(terrain_enu[vis_verts, 3])
    alt_hi = isempty(vis_verts) ? 1000.0 : maximum(terrain_enu[vis_verts, 3])

    # ── Figure ────────────────────────────────────────────────────────────────
    println("Building figure ...")
    set_theme!(theme_latexfonts())
    fig = Figure(size = (1100, 520))

    # ── Panel 1: 300 m top-down ───────────────────────────────────────────────
    ax1 = Axis(fig[1, 1],
               xlabel  = "East  (m)",
               ylabel  = "North  (m)",
               title   = "Top-down  $(round(Int, args["radius"])) m view",
               aspect  = DataAspect())

    # Terrain triangles coloured by mean altitude.
    for i in nearby
        idx  = terrain_faces_raw[i, :] .+ 1
        vx   = terrain_enu[idx, 1]
        vy   = terrain_enu[idx, 2]
        alt  = mean(terrain_enu[idx, 3])
        poly!(ax1, Point2f.(vx, vy);
              color       = alt,
              colormap    = :terrain,
              colorrange  = (alt_lo, alt_hi),
              strokecolor = (:black, 0.3),
              strokewidth = 0.5)
    end

    # Box footprint.
    e_lo, e_hi = extrema(box_enu[:, 1])
    n_lo, n_hi = extrema(box_enu[:, 2])
    box_rect = [Point2f(e_lo, n_lo), Point2f(e_hi, n_lo),
                Point2f(e_hi, n_hi), Point2f(e_lo, n_hi)]
    poly!(ax1, box_rect; color = (:red, 0.0),
          strokecolor = :red, strokewidth = 2)

    # Particle hits.
    sc1 = nothing
    if n_part > 0
        sc1 = scatter!(ax1, part_enu[:, 1], part_enu[:, 2];
                       color      = log_ke,
                       colormap   = :plasma,
                       colorrange = (ke_lo, ke_hi),
                       markersize = 8,
                       strokewidth = 0.5,
                       strokecolor = :black)
    end

    xlims!(ax1, -args["radius"], args["radius"])
    ylims!(ax1, -args["radius"], args["radius"])

    # Terrain colorbar.
    Colorbar(fig[1, 2]; colormap = :terrain, limits = (alt_lo, alt_hi),
             label = "Altitude  (m)", height = Relative(0.8))

    # ── Panel 2: zoom around box ──────────────────────────────────────────────
    z = args["zoom"]
    ax2 = Axis(fig[1, 3],
               xlabel  = "East  (m)",
               ylabel  = "North  (m)",
               title   = "Box zoom  ±$(round(Int,z)) m",
               aspect  = DataAspect())

    # Box triangle edges.
    for face in box_faces_raw
        idx = face .+ 1
        vx  = push!(box_enu[idx, 1], box_enu[idx[1], 1])
        vy  = push!(box_enu[idx, 2], box_enu[idx[1], 2])
        lines!(ax2, vx, vy; color = :red, linewidth = 0.8)
    end

    # Particle hits.
    sc2 = nothing
    if n_part > 0
        sc2 = scatter!(ax2, part_enu[:, 1], part_enu[:, 2];
                       color      = log_ke,
                       colormap   = :plasma,
                       colorrange = (ke_lo, ke_hi),
                       markersize = 10,
                       strokewidth = 0.8,
                       strokecolor = :black)
    end

    xlims!(ax2, -z, z)
    ylims!(ax2, -z, z)

    # Energy colorbar.
    if sc1 !== nothing || sc2 !== nothing
        Colorbar(fig[1, 4]; colormap = :plasma, limits = (ke_lo, ke_hi),
                 label = "log₁₀(KE / GeV)", height = Relative(0.8))
    end

    # Title.
    n_str = n_part > 0 ? "$(n_part) hits" : "no hits"
    Label(fig[0, 1:4],
          "Shower hits — lat=$(round(lat0_deg,digits=3))°  " *
          "lon=$(round(lon0_deg,digits=3))°  ($n_str)",
          fontsize = 13, tellwidth = false)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = dirname(abspath(args["output"]))
    isempty(out_dir) || mkpath(out_dir)
    save(args["output"], fig, px_per_unit = args["dpi"] / 72)
    println("Saved: $(args["output"])")
    println("Done.")
end

main()
