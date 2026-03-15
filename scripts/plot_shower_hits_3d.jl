#!/usr/bin/env julia
"""
3-D shower hit visualisation using GLMakie (OpenGL, correct depth rendering).

Shows the terrain mesh, detector box wireframe, and particle hit positions
in local ENU coordinates.  All three axes use the same metre scale so the
scene is physically proportioned.

Usage
-----
  julia --project=. scripts/plot_shower_hits_3d.jl \\
      TERRAIN.h5 BOX.ply PARTICLES.parquet [options]

  Options:
    --output PATH    Output PNG file          [default: shower_hits_3d.png]
    --radius M       Terrain face centroid radius for display (m)  [default: 200]
    --xlim M         Half-width of all three axes (m)              [default: 50]
    --elev DEG       Camera elevation angle (deg)                  [default: 20]
    --azim DEG       Camera azimuth angle, CW from North (deg)     [default: -60]
    --dpi N          Resolution (dots per inch)                    [default: 200]
    --group STR      HDF5 group for terrain vertices/faces
"""

using ArgParse
using GLMakie
using GLMakie.GeometryBasics: Point3f, TriangleFace, Mesh
using HDF5
using LinearAlgebra
using Parquet2
using Statistics
using Tables

const EARTH_RADIUS_M = 6.371e6

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

function ecef_to_enu(ecef::AbstractMatrix, lat0_deg::Float64, lon0_deg::Float64)
    lat0 = deg2rad(lat0_deg);  lon0 = deg2rad(lon0_deg)
    p0   = EARTH_RADIUS_M .* [cos(lat0)*cos(lon0), cos(lat0)*sin(lon0), sin(lat0)]
    east  = [-sin(lon0),  cos(lon0),  0.0]
    north = [-sin(lat0)*cos(lon0), -sin(lat0)*sin(lon0),  cos(lat0)]
    up    = [ cos(lat0)*cos(lon0),  cos(lat0)*sin(lon0),  sin(lat0)]
    R  = vcat(east', north', up')
    dp = ecef .- p0'
    return (R * dp')'          # (N,3): columns East, North, Up
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
            faces[j] = [idx_reader(io) + 1 for _ in 1:cnt]   # 1-based
        end
        return verts, faces
    end
end

function find_default_group(fid::HDF5.File)::String
    for name in keys(fid)
        g = fid[name]
        isa(g, HDF5.Group) && haskey(g, "vertices") && haskey(g, "faces") && return name
    end
    error("No group with 'vertices' and 'faces' found.")
end

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

function parse_args_main()
    s = ArgParseSettings(description="3-D shower hit plot with GLMakie.",
                         add_help=true)
    @add_arg_table! s begin
        "terrain_h5";  help="Terrain HDF5";         arg_type=String; required=true
        "box_ply";     help="Box PLY";               arg_type=String; required=true
        "particles_parquet"; help="Particles parquet"; arg_type=String; required=true
        "--output";    help="Output PNG";            arg_type=String; default="shower_hits_3d.png"
        "--radius";    help="Terrain display radius (m)"; arg_type=Float64; default=200.0
        "--xlim";      help="Half-width of all axes (m)"; arg_type=Float64; default=50.0
        "--elev";      help="Camera elevation (deg)";     arg_type=Float64; default=20.0
        "--azim";      help="Camera azimuth CW from -Y axis (deg)"; arg_type=Float64; default=-60.0
        "--dpi";       help="Resolution";            arg_type=Int; default=200
        "--group";     help="HDF5 group";            arg_type=String; default=""
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
    terrain_ecef, terrain_faces_raw, lat0, lon0 =
        h5open(args["terrain_h5"], "r") do f
            gname = isempty(args["group"]) ? find_default_group(f) : args["group"]
            println("  Group: $gname");  g = f[gname]
            v   = Float64.(read(g["vertices"]))
            fa  = Int.(read(g["faces"]))
            lat = Float64(read(HDF5.attributes(g)["lat_deg"]))
            lon = Float64(read(HDF5.attributes(g)["lon_deg"]))
            v, fa, lat, lon
        end
    println("  $(size(terrain_ecef,1)) vertices, $(size(terrain_faces_raw,1)) faces")

    # ── Load box ──────────────────────────────────────────────────────────────
    println("Reading box: $(args["box_ply"])")
    box_ecef, box_faces_raw = read_ply(args["box_ply"])
    println("  $(size(box_ecef,1)) vertices, $(length(box_faces_raw)) faces")

    # ── Load particles ────────────────────────────────────────────────────────
    println("Reading particles: $(args["particles_parquet"])")
    ds   = Parquet2.Dataset(args["particles_parquet"])
    raw  = Tables.columntable(ds)
    cols = propertynames(raw)
    function getcol(names...)
        for n in names
            sym = Symbol(n)
            sym in cols && return Vector{Float64}(getproperty(raw, sym))
        end
        error("None of $(names) found in columns: $(cols)")
    end
    px_loc = getcol("x", "position_x")
    py_loc = getcol("y", "position_y")
    pz_loc = getcol("z", "position_z")
    ke_raw = getcol("kinetic_energy", "energy", "ekin")
    # unit check: if median > 1e6 assume eV
    ke     = median(ke_raw) > 1e6 ? ke_raw ./ 1e9 : ke_raw
    pdg_sym = findfirst(n -> n in cols, [Symbol("pdg_code"), Symbol("pdg"),
                                          Symbol("particle")])
    pdg = pdg_sym !== nothing ? Vector{Int}(getproperty(raw, cols[pdg_sym])) :
                                fill(0, length(px_loc))
    n_part = length(px_loc)
    println("  $n_part particle hits")

    # ── Convert to ENU ────────────────────────────────────────────────────────
    terrain_enu = ecef_to_enu(terrain_ecef, lat0, lon0)
    box_enu     = ecef_to_enu(box_ecef,     lat0, lon0)
    box_cen     = vec(mean(box_enu, dims=1))
    println("  Box centroid ENU: East=$(round(box_cen[1],digits=1)) m, " *
            "North=$(round(box_cen[2],digits=1)) m, " *
            "Alt=$(round(box_cen[3],digits=1)) m")

    # Particle positions: ECEF offsets from box centroid → rotate to ENU
    lat0r = deg2rad(lat0);  lon0r = deg2rad(lon0)
    east_hat  = [-sin(lon0r),  cos(lon0r),  0.0]
    north_hat = [-sin(lat0r)*cos(lon0r), -sin(lat0r)*sin(lon0r),  cos(lat0r)]
    up_hat    = [ cos(lat0r)*cos(lon0r),  cos(lat0r)*sin(lon0r),  sin(lat0r)]
    R_enu     = vcat(east_hat', north_hat', up_hat')

    part_xyz_off = hcat(px_loc, py_loc, pz_loc)        # (N,3) ECEF offsets
    enu_off      = (R_enu * part_xyz_off')'            # (N,3) ENU offsets
    part_east    = enu_off[:, 1] .+ box_cen[1]
    part_north   = enu_off[:, 2] .+ box_cen[2]
    part_up      = enu_off[:, 3] .+ box_cen[3]

    log_ke = log10.(clamp.(ke, 1e-4, Inf))
    ke_lo  = minimum(log_ke);  ke_hi = maximum(log_ke)

    # ── Filter terrain faces ──────────────────────────────────────────────────
    xl = args["xlim"]
    face_cen_e = [mean(terrain_enu[terrain_faces_raw[i,:] .+ 1, 1]) for i in 1:size(terrain_faces_raw,1)]
    face_cen_n = [mean(terrain_enu[terrain_faces_raw[i,:] .+ 1, 2]) for i in 1:size(terrain_faces_raw,1)]
    nearby = findall(hypot.(face_cen_e, face_cen_n) .<= args["radius"])
    println("  $(length(nearby)) terrain faces within $(args["radius"]) m")

    # ── Interpolate terrain altitude at box centre (ENU origin) ───────────────
    # Find the face with centroid closest to origin, then barycentric interpolation.
    face_dist2 = face_cen_e.^2 .+ face_cen_n.^2
    cf  = argmin(face_dist2)
    ci  = terrain_faces_raw[cf, :] .+ 1          # 1-based vertex indices
    v0  = terrain_enu[ci[1], :]; v1 = terrain_enu[ci[2], :]; v2 = terrain_enu[ci[3], :]
    den = (v1[2]-v2[2])*(v0[1]-v2[1]) + (v2[1]-v1[1])*(v0[2]-v2[2])
    l0  = ((v1[2]-v2[2])*(0-v2[1]) + (v2[1]-v1[1])*(0-v2[2])) / den
    l1  = ((v2[2]-v0[2])*(0-v2[1]) + (v0[1]-v2[1])*(0-v2[2])) / den
    l2  = 1 - l0 - l1
    terrain_alt_origin = l0*v0[3] + l1*v1[3] + l2*v2[3]
    println("  Terrain altitude at box centre (interpolated): $(round(terrain_alt_origin,digits=1)) m")
    box_bottom = minimum(box_enu[:, 3])
    println("  Box bottom: $(round(box_bottom,digits=1)) m  " *
            "(gap = $(round(box_bottom - terrain_alt_origin, digits=1)) m)")

    # Shift box and particles so box bottom sits on the interpolated terrain surface.
    z_shift = terrain_alt_origin - box_bottom
    box_enu[:, 3] .+= z_shift
    box_cen = vec(mean(box_enu, dims=1))
    part_up .+= z_shift

    vis_verts = sort(unique(vcat([terrain_faces_raw[i,:] .+ 1 for i in nearby]...)))
    alt_lo = minimum(terrain_enu[vis_verts, 3])
    alt_hi = maximum(terrain_enu[vis_verts, 3])
    remap  = Dict(v => i for (i,v) in enumerate(vis_verts))

    # ── Build terrain Mesh ────────────────────────────────────────────────────
    mesh_verts = [Point3f(terrain_enu[v,1], terrain_enu[v,2], terrain_enu[v,3])
                  for v in vis_verts]
    mesh_faces = [TriangleFace(remap[terrain_faces_raw[i,1]+1],
                               remap[terrain_faces_raw[i,2]+1],
                               remap[terrain_faces_raw[i,3]+1])
                  for i in nearby]
    terrain_mesh = Mesh(mesh_verts, mesh_faces)
    vert_alts    = Float32[terrain_enu[v, 3] for v in vis_verts]

    # ── Figure ────────────────────────────────────────────────────────────────
    println("Building figure ...")
    GLMakie.activate!(; title="Shower hits 3D")
    set_theme!(theme_latexfonts())

    fig = Figure(size=(1100, 900))

    # Axis3 with equal data-unit scaling
    ax = Axis3(fig[1, 1];
               xlabel = "East  (m)",
               ylabel = "North  (m)",
               zlabel = "Altitude  (m)",
               title  = "3-D shower hits — lat=$(round(lat0,digits=3))°  " *
                         "lon=$(round(lon0,digits=3))°  ($n_part hits)",
               aspect = (1, 1, 1),
               viewmode = :fit,
               limits = (-xl, xl, -xl, xl, terrain_alt_origin - 2.0, terrain_alt_origin + 2*xl))

    # ── Terrain ───────────────────────────────────────────────────────────────
    ms = mesh!(ax, terrain_mesh;
               color      = vert_alts,
               colormap   = :terrain,
               colorrange = (Float32(alt_lo), Float32(alt_hi)),
               alpha      = 0.85,
               transparency = false)
    Colorbar(fig[1, 2]; colormap=:terrain, limits=(alt_lo, alt_hi),
             label="Altitude  (m)", height=Relative(0.7))

    # ── Box wireframe ─────────────────────────────────────────────────────────
    for face in box_faces_raw
        idx = [face; face[1]]
        vx  = box_enu[idx, 1]
        vy  = box_enu[idx, 2]
        vz  = box_enu[idx, 3]
        lines!(ax, vx, vy, vz; color=:red, linewidth=1.0)
    end

    # ── Particle scatter ──────────────────────────────────────────────────────
    pdg_codes  = sort(unique(pdg))
    markers_jl = Dict(22 => :circle, 11 => :rect, -11 => :rect,
                       13 => :utriangle, -13 => :utriangle)
    labels_jl  = Dict(22 => "photon", 11 => "e⁻", -11 => "e⁺",
                       13 => "μ⁻",    -13 => "μ⁺")

    sc = nothing
    for code in pdg_codes
        mask = pdg .== code
        sc = scatter!(ax,
                      part_east[mask], part_north[mask], part_up[mask];
                      color      = log_ke[mask],
                      colormap   = :plasma,
                      colorrange = (ke_lo, ke_hi),
                      markersize = 10,
                      marker     = get(markers_jl, code, :circle),
                      strokewidth = 0.5,
                      strokecolor = :black,
                      label      = get(labels_jl, code, string(code)))
    end
    if sc !== nothing
        axislegend(ax; position=:lt, labelsize=10)
        Colorbar(fig[1, 3]; colormap=:plasma, limits=(ke_lo, ke_hi),
                 label="log₁₀(KE / GeV)", height=Relative(0.7))
    end

    # ── Camera ────────────────────────────────────────────────────────────────
    # Convert elev/azim to Makie's spherical camera angles
    elev_rad = deg2rad(args["elev"])
    azim_rad = deg2rad(args["azim"])
    ax.elevation[] = elev_rad
    ax.azimuth[]   = azim_rad

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = dirname(abspath(args["output"]))
    isempty(out_dir) || mkpath(out_dir)
    save(args["output"], fig; px_per_unit = args["dpi"] / 72)
    println("Saved: $(args["output"])")
end

main()
