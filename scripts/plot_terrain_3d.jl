#!/usr/bin/env julia
"""
Two-panel 3D terrain + detector box mesh plot in local ENU coordinates.

Panel 1  Terrain triangles within r1 horizontal distance (default 10 000 m)
Panel 2  Terrain triangles within r2 horizontal distance (default 100 m)

The terrain is coloured by altitude using the same :terrain colormap as the
2-D scatter plot.  The detector box is overlaid in red.

Usage
-----
  julia --project=<dir> plot_terrain_3d.jl TERRAIN.h5 BOX.ply [options]

  Options:
    --group  NAME   HDF5 group for terrain  [default: first group with vertices/faces]
    --r1     R      Outer panel radius (m)  [default: 10000]
    --r2     R      Inner panel radius (m)  [default: 100]
    --output PATH   Output PNG file         [default: terrain_3d.png]
    --dpi    N      Figure resolution       [default: 150]
"""

using ArgParse
using CairoMakie
using HDF5
using LinearAlgebra
using Statistics

const EARTH_RADIUS_M = 6.371e6

# ---------------------------------------------------------------------------
# Coordinate conversion  (identical to plot_terrain_box.jl)
# ---------------------------------------------------------------------------

function ecef_to_enu(ecef::AbstractMatrix, lat0_deg::Float64, lon0_deg::Float64)
    lat0 = deg2rad(lat0_deg)
    lon0 = deg2rad(lon0_deg)
    p0   = EARTH_RADIUS_M .* [cos(lat0)*cos(lon0), cos(lat0)*sin(lon0), sin(lat0)]
    east  = [-sin(lon0),  cos(lon0),  0.0]
    north = [-sin(lat0)*cos(lon0), -sin(lat0)*sin(lon0),  cos(lat0)]
    up    = [ cos(lat0)*cos(lon0),  cos(lat0)*sin(lon0),  sin(lat0)]
    R  = vcat(east', north', up')   # 3×3
    dp = ecef .- p0'                # (N,3)
    return (R * dp')'               # (N,3)
end

# ---------------------------------------------------------------------------
# PLY reader  (identical to plot_terrain_box.jl)
# ---------------------------------------------------------------------------

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
            tokens = split(line)
            isempty(tokens) && continue
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
# HDF5 group helper
# ---------------------------------------------------------------------------

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
# Mesh filtering
# ---------------------------------------------------------------------------

"""
Filter terrain to triangles whose vertices all lie within `radius` metres
of the origin in the horizontal (East–North) plane.

`faces_0based` is (M, 3) with 0-based indices into `enu`.

Returns `(enu_filtered, faces_1based)` where `faces_1based` is (M', 3)
with 1-based indices into `enu_filtered`.  Returns nothing if no faces pass.
"""
function filter_mesh_horiz(enu::AbstractMatrix, faces_0based::AbstractMatrix,
                           radius::Float64)
    horiz  = sqrt.(enu[:, 1].^2 .+ enu[:, 2].^2)
    in_reg = horiz .<= radius
    old_idx = findall(in_reg)
    isempty(old_idx) && return enu[1:0, :], zeros(Int, 0, 3)

    new_idx = zeros(Int, size(enu, 1))
    new_idx[old_idx] .= 1:length(old_idx)
    enu_f = enu[old_idx, :]

    # Convert 0-based → 1-based for index lookup, then filter faces.
    fa = faces_0based[:, 1] .+ 1
    fb = faces_0based[:, 2] .+ 1
    fc = faces_0based[:, 3] .+ 1
    mask = (new_idx[fa] .!= 0) .& (new_idx[fb] .!= 0) .& (new_idx[fc] .!= 0)

    faces_f = hcat(new_idx[fa[mask]], new_idx[fb[mask]], new_idx[fc[mask]])  # (M',3) 1-based
    return enu_f, faces_f
end

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

function parse_args_main()
    s = ArgParseSettings(
        description = "3-D terrain + box mesh plot in local ENU coordinates.",
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
        "--r1"
            help     = "Outer panel horizontal radius (m)"
            arg_type = Float64
            default  = 10000.0
        "--r2"
            help     = "Inner panel horizontal radius (m)"
            arg_type = Float64
            default  = 100.0
        "--output"
            help     = "Output PNG file path"
            arg_type = String
            default  = "terrain_3d.png"
        "--dpi"
            help     = "Figure resolution (dots per inch)"
            arg_type = Int
            default  = 250
    end
    return ArgParse.parse_args(s)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    args = parse_args_main()

    # ── Load terrain ──────────────────────────────────────────────────────────
    println("Reading terrain: $(args["terrain_h5"])")
    terrain_verts_ecef, terrain_faces_0based, lat0_deg, lon0_deg =
        h5open(args["terrain_h5"], "r") do f
            gname = isempty(args["group"]) ? find_default_group(f) : args["group"]
            println("  Group: $gname")
            g = f[gname]
            v = Float64.(read(g["vertices"]))   # (N, 3) ECEF
            fc = Int.(read(g["faces"]))          # (M, 3) 0-based
            lat0 = Float64(read(HDF5.attributes(g)["lat_deg"]))
            lon0 = Float64(read(HDF5.attributes(g)["lon_deg"]))
            v, fc, lat0, lon0
        end
    println("  $(size(terrain_verts_ecef, 1)) vertices, " *
            "$(size(terrain_faces_0based, 1)) faces  " *
            "(lat=$(round(lat0_deg,digits=4))°  lon=$(round(lon0_deg,digits=4))°)")

    # ── Load box ──────────────────────────────────────────────────────────────
    println("Reading box: $(args["box_ply"])")
    box_verts_ecef, box_faces_raw = read_ply(args["box_ply"])
    # box faces from PLY are 0-based; convert to (M,3) 1-based matrix.
    box_faces_1based = hcat([f .+ 1 for f in box_faces_raw]...)' |> Matrix{Int}
    println("  $(size(box_verts_ecef, 1)) vertices, $(length(box_faces_raw)) faces")

    # ── Project to ENU ────────────────────────────────────────────────────────
    terrain_enu = ecef_to_enu(terrain_verts_ecef, lat0_deg, lon0_deg)
    box_enu     = ecef_to_enu(box_verts_ecef,     lat0_deg, lon0_deg)

    # ── Snap box to terrain surface ───────────────────────────────────────────
    # Find terrain altitude at the box centre (E=0, N=0) via closest vertex.
    horiz_to_centre = sqrt.(terrain_enu[:, 1].^2 .+ terrain_enu[:, 2].^2)
    terrain_alt_at_centre = terrain_enu[argmin(horiz_to_centre), 3]
    box_bottom = minimum(box_enu[:, 3])
    snap_dz    = terrain_alt_at_centre - box_bottom
    box_enu_snapped = copy(box_enu)
    box_enu_snapped[:, 3] .+= snap_dz
    println("  Snapping box: terrain surface at centre = " *
            "$(round(terrain_alt_at_centre, digits=1)) m  " *
            "(shift = $(round(snap_dz, digits=1)) m)")

    # Shared altitude colour limits from the full terrain.
    alt_lo, alt_hi = extrema(terrain_enu[:, 3])
    clims = (alt_lo, alt_hi)
    cmap  = :terrain

    # Box vertices/faces as plain matrices for Makie's mesh! API.
    box_verts_f32 = Float32.(box_enu_snapped)  # (N, 3)
    # box_faces_1based is already (M, 3) 1-based Int matrix.

    # ── Figure ────────────────────────────────────────────────────────────────
    println("Building figure ...")
    set_theme!(theme_latexfonts())
    fig = Figure(size = (1200, 420), figure_padding = (2, 10, 2, 2))

    radii  = [args["r1"], args["r2"]]
    labels = ["Within $(round(Int, args["r1"]/1e3)) km", "Within $(round(Int, args["r2"])) m"]

    local sc_ref   # for the colourbar

    for (col, (radius, label)) in enumerate(zip(radii, labels))
        enu_f, faces_f = filter_mesh_horiz(terrain_enu, terrain_faces_0based, radius)
        n_v = size(enu_f, 1)
        n_f = size(faces_f, 1)
        println("  Panel $col ($label): $n_v vertices, $n_f faces")

        ax = Axis3(fig[1, col],
                   title   = label,
                   xlabel  = "East (m)",
                   ylabel  = "North (m)",
                   zlabel  = "Altitude (m)",
                   titlesize = 13,
                   aspect  = (1, 1, 1))

        if n_f > 0
            verts_f32 = Float32.(enu_f)   # (N, 3)
            alts      = enu_f[:, 3]

            sc_ref = mesh!(ax, verts_f32, faces_f;
                           color      = Float32.(alts),
                           colormap   = cmap,
                           colorrange = clims,
                           shading    = true)
        end

        # Box mesh.
        mesh!(ax, box_verts_f32, box_faces_1based;
              color = (:red, 0.75), shading = NoShading)

        # Set physically proportional axis limits: z spans same range as x/y.
        half = radius * 1.05
        xlims!(ax, -half, half)
        ylims!(ax, -half, half)
        zlims!(ax, terrain_alt_at_centre - half, terrain_alt_at_centre + half)
    end

    # Shared colourbar.
    if @isdefined(sc_ref)
        Colorbar(fig[1, 3], sc_ref,
                 label  = "Altitude (m)",
                 height = Relative(0.80))
    end

    colgap!(fig.layout, 1, -80)   # pull the two 3-D panels together
    colgap!(fig.layout, 2,  10)   # small gap before colourbar

    Label(fig[0, 1:2],
          "Terrain mesh + detector box  —  " *
          "lat=$(round(lat0_deg,digits=3))°  lon=$(round(lon0_deg,digits=3))°",
          fontsize = 13, tellwidth = false)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = dirname(abspath(args["output"]))
    isempty(out_dir) || mkpath(out_dir)
    save(args["output"], fig, px_per_unit = args["dpi"] / 72)
    println("Saved: $(args["output"])")
    println("Done.")
end

main()
