#!/usr/bin/env julia
"""
Create a rectangular box observation mesh in CORSIKA geocentric world coordinates.

The box is defined in a local East-North-Up (ENU) frame at the target
latitude/longitude, then rotated into the CORSIKA geocentric ECEF frame.

CORSIKA geocentric frame
  +x  lat=0, lon=0   (prime meridian on equator)
  +y  lat=0, lon=90E
  +z  geographic north pole

Local ENU frame at (lat, lon)
  x  east
  y  north
  z  up (radially outward)

Yaw convention
  --yaw rotates the box CCW (viewed from above) about the local up axis.
  yaw=0 means the box length axis (local x) points east.

Usage
-----
  julia --project=<project_dir> make_box_scene.jl [options]

  Required:
    --lat     LAT     Target latitude  (degrees, -90..90)
    --lon     LON     Target longitude (degrees, -180..180)
    --length  L       Box length along local east  axis (metres)
    --width   W       Box width  along local north axis (metres)
    --height  H       Box height along local up    axis (metres)

  Optional:
    --terrain-h5 PATH Path to terrain HDF5 (from create_terrain_geometry.jl).
                      When provided, the box bottom altitude is derived from
                      the terrain mesh at the target lat/lon.  Overrides
                      --altitude.
    --altitude A      Altitude of box bottom above Earth surface  [default: 0]
                      Ignored when --terrain-h5 is given.
    --yaw      Y      Yaw angle, CCW from east (degrees)          [default: 0]
    --nx       NX     Grid subdivisions along length              [default: 4]
    --ny       NY     Grid subdivisions along width               [default: 4]
    --nz       NZ     Grid subdivisions along height              [default: 2]
    --output   PATH   Output PLY file                            [default: box.ply]
"""

using ArgParse
using HDF5
using LinearAlgebra

const EARTH_RADIUS_M = 6.371e6

# ---------------------------------------------------------------------------
# Geographic rotation
# ---------------------------------------------------------------------------

"""
Return the 3×3 rotation matrix that maps local ENU coordinates at (lat_deg,
lon_deg) into the CORSIKA geocentric ECEF frame.

The local x/y axes are first rotated CCW by yaw_deg about the local up axis,
then the geographic rotation is applied.  yaw=0 leaves local-x pointing east.

Returns R such that ecef_vec = R * enu_vec.
"""
function rotation_matrix_geo(lat_deg::Float64, lon_deg::Float64,
                              yaw_deg::Float64 = 0.0)
    phi   = deg2rad(lat_deg)
    lam   = deg2rad(lon_deg)
    theta = deg2rad(yaw_deg)

    east  = [-sin(lam),  cos(lam),  0.0]
    north = [-sin(phi)*cos(lam), -sin(phi)*sin(lam),  cos(phi)]
    up    = [ cos(phi)*cos(lam),  cos(phi)*sin(lam),  sin(phi)]

    local_x = cos(theta) .* east  .- sin(theta) .* north
    local_y = sin(theta) .* east  .+ cos(theta) .* north

    hcat(local_x, local_y, up)   # 3 × 3, columns are local basis in ECEF
end

# ---------------------------------------------------------------------------
# Terrain altitude lookup
# ---------------------------------------------------------------------------

"""
Return the terrain altitude (metres above Earth reference sphere) at the
target (lat_deg, lon_deg) by barycentric interpolation of the terrain mesh.

Finds the face whose centroid is closest to the target in the horizontal
(East-North) plane, then interpolates the altitude at (E=0, N=0) using
barycentric coordinates.
"""
function terrain_altitude_at_target(terrain_h5::String,
                                    lat_deg::Float64, lon_deg::Float64)
    lat0 = deg2rad(lat_deg)
    lon0 = deg2rad(lon_deg)

    p0    = EARTH_RADIUS_M .* [cos(lat0)*cos(lon0), cos(lat0)*sin(lon0), sin(lat0)]
    east  = [-sin(lon0),  cos(lon0),  0.0]
    north = [-sin(lat0)*cos(lon0), -sin(lat0)*sin(lon0),  cos(lat0)]
    up    = [ cos(lat0)*cos(lon0),  cos(lat0)*sin(lon0),  sin(lat0)]
    R     = vcat(east', north', up')   # 3×3

    h5open(terrain_h5, "r") do f
        gname = ""
        for name in keys(f)
            g = f[name]
            if isa(g, HDF5.Group) && haskey(g, "vertices")
                gname = name
                break
            end
        end
        isempty(gname) && error("No group with 'vertices' found in $terrain_h5")

        verts = Float64.(read(f[gname]["vertices"]))   # (N, 3) ECEF
        faces = Int.(read(f[gname]["faces"]))          # (M, 3) 0-based
        enu   = (R * (verts .- p0')')'                 # (N, 3) ENU

        # Find the face with centroid closest to the target (E=0, N=0).
        n_faces = size(faces, 1)
        best_dist2 = Inf
        best_face  = 1
        for i in 1:n_faces
            i0, i1, i2 = faces[i, 1]+1, faces[i, 2]+1, faces[i, 3]+1
            ce = (enu[i0,1] + enu[i1,1] + enu[i2,1]) / 3.0
            cn = (enu[i0,2] + enu[i1,2] + enu[i2,2]) / 3.0
            d2 = ce^2 + cn^2
            if d2 < best_dist2
                best_dist2 = d2
                best_face  = i
            end
        end

        # Barycentric interpolation at (E=0, N=0) within the closest face.
        i0, i1, i2 = faces[best_face,1]+1, faces[best_face,2]+1, faces[best_face,3]+1
        e0, n0, u0 = enu[i0, 1], enu[i0, 2], enu[i0, 3]
        e1, n1, u1 = enu[i1, 1], enu[i1, 2], enu[i1, 3]
        e2, n2, u2 = enu[i2, 1], enu[i2, 2], enu[i2, 3]
        denom = (n1 - n2)*(e0 - e2) + (e2 - e1)*(n0 - n2)
        l0    = ((n1 - n2)*(0 - e2) + (e2 - e1)*(0 - n2)) / denom
        l1    = ((n2 - n0)*(0 - e2) + (e0 - e2)*(0 - n2)) / denom
        l2    = 1.0 - l0 - l1
        alt   = l0*u0 + l1*u1 + l2*u2
        println("  Terrain interpolation: face $best_face, " *
                "centroid dist=$(round(sqrt(best_dist2),digits=1)) m, " *
                "altitude=$(round(alt,digits=2)) m")
        return alt
    end
end

# ---------------------------------------------------------------------------
# Box mesh generator
# ---------------------------------------------------------------------------

"""
Tessellate a bilinear quadrilateral into an na × nb grid of triangles.
"""
function grid_face(corners::AbstractVector, na::Int, nb::Int)
    c0, c1, c2, c3 = corners
    verts = Vector{Vector{Float64}}()
    for j in 0:nb, i in 0:na
        s, t = i / na, j / nb
        push!(verts, (1-s)*(1-t)*c0 + s*(1-t)*c1 + s*t*c2 + (1-s)*t*c3)
    end

    tris = Vector{Tuple{Int,Int,Int}}()
    stride = na + 1
    for j in 0:(nb-1), i in 0:(na-1)
        v00 = j*stride + i + 1
        v10 = j*stride + i + 2
        v01 = (j+1)*stride + i + 1
        v11 = (j+1)*stride + i + 2
        push!(tris, (v00, v10, v11))
        push!(tris, (v00, v11, v01))
    end
    verts, tris
end

"""
Build a closed rectangular box mesh in local ENU coordinates.

The box is horizontally centred at (0, 0) with its bottom face at z = 0.

Returns (vertices, faces) where faces are 0-based index triplets.
"""
function make_box_mesh(length_::Float64, width_::Float64, height_::Float64,
                       nx::Int, ny::Int, nz::Int)
    hl, hw = length_ / 2.0, width_ / 2.0
    h = height_

    all_verts = Vector{Vector{Float64}}()
    all_faces = Vector{Tuple{Int,Int,Int}}()

    function add_face!(corners, na, nb)
        offset = length(all_verts)
        fv, ft = grid_face(corners, na, nb)
        append!(all_verts, fv)
        for (a, b, c) in ft
            push!(all_faces, (a + offset - 1, b + offset - 1, c + offset - 1))
        end
    end

    # Six faces with outward-pointing normals (CCW winding).
    add_face!([[-hl,-hw,0.], [ hl,-hw,0.], [ hl, hw,0.], [-hl, hw,0.]], nx, ny)  # bottom
    add_face!([[-hl,-hw,h],  [-hl, hw,h],  [ hl, hw,h],  [ hl,-hw,h]],  nx, ny)  # top
    add_face!([[-hl,-hw,0.], [ hl,-hw,0.], [ hl,-hw,h],  [-hl,-hw,h]],  nx, nz)  # south
    add_face!([[ hl, hw,0.], [-hl, hw,0.], [-hl, hw,h],  [ hl, hw,h]],  nx, nz)  # north
    add_face!([[-hl, hw,0.], [-hl,-hw,0.], [-hl,-hw,h],  [-hl, hw,h]],  ny, nz)  # west
    add_face!([[ hl,-hw,0.], [ hl, hw,0.], [ hl, hw,h],  [ hl,-hw,h]],  ny, nz)  # east

    all_verts, all_faces
end

# ---------------------------------------------------------------------------
# PLY writer
# ---------------------------------------------------------------------------

function write_ply_binary(filepath::AbstractString,
                          vertices::AbstractVector,
                          faces::AbstractVector)
    n_verts = length(vertices)
    n_faces = length(faces)

    header = "ply\n" *
             "format binary_little_endian 1.0\n" *
             "comment generated by make_box_scene.jl\n" *
             "element vertex $(n_verts)\n" *
             "property double x\n" *
             "property double y\n" *
             "property double z\n" *
             "element face $(n_faces)\n" *
             "property list uchar uint vertex_indices\n" *
             "end_header\n"

    open(filepath, "w") do io
        write(io, header)
        for v in vertices
            write(io, htol(Float64(v[1])))
            write(io, htol(Float64(v[2])))
            write(io, htol(Float64(v[3])))
        end
        for (a, b, c) in faces
            write(io, UInt8(3))
            write(io, htol(UInt32(a)))
            write(io, htol(UInt32(b)))
            write(io, htol(UInt32(c)))
        end
    end

    println("Wrote $(n_verts) vertices, $(n_faces) triangles " *
            "-> $(round(filesize(filepath)/1e3, digits=1)) kB")
end

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

function parse_args_main()
    s = ArgParseSettings(
        description = "Create a rectangular box observation mesh for CORSIKA.",
        add_help    = true,
    )
    @add_arg_table! s begin
        "--lat"
            help     = "Target latitude  (degrees, -90..90)"
            arg_type = Float64
            required = true
        "--lon"
            help     = "Target longitude (degrees, -180..180)"
            arg_type = Float64
            required = true
        "--length"
            help     = "Box length along local east  axis (metres)"
            arg_type = Float64
            required = true
        "--width"
            help     = "Box width  along local north axis (metres)"
            arg_type = Float64
            required = true
        "--height"
            help     = "Box height along local up    axis (metres)"
            arg_type = Float64
            required = true
        "--terrain-h5"
            help     = "Terrain HDF5 file; altitude is derived from the mesh at the target lat/lon"
            arg_type = String
            default  = ""
        "--altitude"
            help     = "Altitude of box bottom above Earth surface (metres); ignored when --terrain-h5 is given"
            arg_type = Float64
            default  = 0.0
        "--yaw"
            help     = "Yaw: CCW rotation about local up axis (degrees); 0 = east"
            arg_type = Float64
            default  = 0.0
        "--nx"
            help     = "Grid subdivisions along length"
            arg_type = Int
            default  = 4
        "--ny"
            help     = "Grid subdivisions along width"
            arg_type = Int
            default  = 4
        "--nz"
            help     = "Grid subdivisions along height"
            arg_type = Int
            default  = 2
        "--output"
            help     = "Output PLY file"
            arg_type = String
            default  = "box.ply"
    end
    return ArgParse.parse_args(s)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    args = parse_args_main()

    lat_deg = Float64(args["lat"])
    lon_deg = Float64(args["lon"])

    if !isempty(args["terrain-h5"])
        altitude = terrain_altitude_at_target(args["terrain-h5"], lat_deg, lon_deg)
        println("Altitude from terrain mesh: $(round(altitude, digits=1)) m  " *
                "($(args["terrain-h5"]))")
    else
        altitude = Float64(args["altitude"])
    end

    verts_local, faces = make_box_mesh(
        Float64(args["length"]), Float64(args["width"]), Float64(args["height"]),
        args["nx"], args["ny"], args["nz"],
    )

    z_offset = EARTH_RADIUS_M + altitude
    verts_translated = [v .+ [0.0, 0.0, z_offset] for v in verts_local]

    R = rotation_matrix_geo(lat_deg, lon_deg, Float64(args["yaw"]))
    verts_world = [R * v for v in verts_translated]

    centroid = sum(verts_world) ./ length(verts_world)
    println("Centroid altitude:   $(round(norm(centroid) - EARTH_RADIUS_M, digits=1)) m")

    out_dir = dirname(abspath(args["output"]))
    isempty(out_dir) || mkpath(out_dir)

    write_ply_binary(args["output"], verts_world, faces)
end

main()
