# CORSIKA Terrain Geometry

Scripts to build a terrain mesh and rectangular detector box for CORSIKA 8
air-shower simulations, plus visualisation tools to verify the geometry.

---

## Contents

```
scripts/
  create_terrain_geometry.jl  – HDF5 terrain mesh from a global triangulation
  terrain_h5_to_ply.jl        – Convert terrain HDF5 → CORSIKA-compatible PLY
  make_box_scene.jl           – Rectangular detector box PLY
  plot_terrain_box.jl         – 2-D (top-down + cross-section) verification plot
  plot_terrain_3d.jl          – 3-D mesh visualisation (two-panel)
applications/
  terrain_shower.cpp          – CORSIKA 8 C++ application
  CMakeLists.txt
resources/
  wc2.1_10m_elev.tif          – 10 arcmin WorldClim elevation reference (1.3 MB)
```

---

## Prerequisites

### Julia

Install Julia via [juliaup](https://github.com/JuliaLang/juliaup) (recommended):

```bash
curl -fsSL https://install.julialang.org | sh
```

Then open a new terminal and confirm:

```bash
julia --version   # should print julia version 1.x.x
```

### Large data files (not distributed)

Two large HDF5 files are required and must be set via environment variables.
They are not included in this repository.

| Variable                        | File                   | Typical size |
|---------------------------------|------------------------|-------------|
| `CORSIKA_TERRAIN_TRIANGULATION` | `triangulation.h5`     | ~13 GB      |
| `CORSIKA_TERRAIN_ELEVATION`     | `earth_relief_15s.h5`  | ~14 GB      |

Set these in your shell (add to `~/.zshrc` or `~/.bashrc` to make permanent):

```bash
export CORSIKA_TERRAIN_TRIANGULATION=/path/to/triangulation.h5
export CORSIKA_TERRAIN_ELEVATION=/path/to/earth_relief_15s.h5
```

---

## Setup

Navigate to this directory and install all Julia dependencies.
This only needs to be done once:

```bash
cd /path/to/CORSIKA_terrain
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This reads `Project.toml` and downloads all required packages into an isolated
environment (no global packages are modified).

---

## Workflow

All scripts are run from the `CORSIKA_terrain` directory with
`julia --project=. scripts/<name>.jl`.  Pass `--help` to any script for a
full list of options.

### Step 1 – Create terrain geometry

Loads the base sphere triangulation, rotates it to the target location,
evaluates terrain elevation at each vertex, and writes an HDF5 file
containing ECEF vertex coordinates and triangle faces.

```bash
julia --project=. scripts/create_terrain_geometry.jl LAT LON \
    --group base_triangulation_100000 \
    --max-angle 0.05 \
    --output terrain.h5
```

| Option | Description | Default |
|--------|-------------|---------|
| `LAT` | Target latitude (degrees, positional) | required |
| `LON` | Target longitude (degrees, positional) | required |
| `--group` | Triangulation density group in `triangulation.h5`. Denser groups give finer meshes but are slower. Available: `base_triangulation_1000`, `_3000`, `_10000`, `_30000`, `_100000`, `_300000` | `base_triangulation_30000` |
| `--output` | Output HDF5 file | `terrain.h5` |

**Example** (Jura region, France):

```bash
julia --project=. scripts/create_terrain_geometry.jl 46.34249442 5.83882331 \
    --group base_triangulation_100000 \
    --output terrain.h5
```

Expected output:
```
Loaded 300000 vertices, 599996 faces (1-based)
Using all 300000 vertices, 599996 faces
Evaluated elevation over 648 tiles
Elevation range: -418 .. 8727 m  (mean: 231 m)
Wrote 300000 vertices, 599996 faces
```

### Step 2 – Convert terrain HDF5 to PLY

Converts the terrain geometry to binary PLY format for CORSIKA's
`MeshLoader::loadPLY`.  Vertices are written as double-precision (Float64).

```bash
julia --project=. scripts/terrain_h5_to_ply.jl terrain.h5 \
    --output terrain.ply
```

Expected output:
```
Vertices: 30951   Faces: 61227
Terrain altitude: 359 .. 1244 m  (mean 839 m)
Wrote 30951 vertices, 61227 faces -> 1.5 MB
```

### Step 3 – Create detector box

Builds a closed rectangular box mesh in ECEF coordinates and writes a binary
PLY file.

The box is defined in a local East-North-Up (ENU) frame at the target
lat/lon, then rotated into CORSIKA's geocentric frame.

```bash
julia --project=. scripts/make_box_scene.jl \
    --lat LAT --lon LON \
    --length L --width W --height H \
    --yaw Y \
    --output box.ply
```

| Option | Description | Default |
|--------|-------------|---------|
| `--lat`, `--lon` | Target location (degrees) | required |
| `--length` | Box length along local East axis (m) | required |
| `--width` | Box width along local North axis (m) | required |
| `--height` | Box height along local Up axis (m) | required |
| `--terrain-h5` | Terrain HDF5 file (from Step 1). When given, the box bottom altitude is read from the terrain mesh at the target lat/lon. **Recommended** — avoids having to specify the altitude manually. | `""` |
| `--altitude` | Height of box bottom above Earth surface (m). Ignored when `--terrain-h5` is given. | `0` |
| `--yaw` | Rotation CCW from East (degrees); 0 = length points East | `0` |
| `--output` | Output PLY file | `box.ply` |

**Example** (same location, 12.2 × 2.44 × 2.59 m box, yaw 90°):

```bash
julia --project=. scripts/make_box_scene.jl \
    --lat 46.34249442 --lon 5.83882331 \
    --length 12.2 --width 2.44 --height 2.59 \
    --terrain-h5 terrain.h5 --yaw 90 \
    --output box.ply
```

Expected output:
```
Altitude from terrain mesh: 658.5 m  (terrain.h5)
Centroid altitude:   659.8 m
Wrote 110 vertices, 128 triangles -> 4.5 kB
```

### Step 4 – 2-D verification plot

Three-panel figure (top-down, east cross-section, north cross-section) in
local ENU coordinates.  Useful for a quick sanity check that the terrain
and box are in the right place.

```bash
julia --project=. scripts/plot_terrain_box.jl terrain.h5 box.ply \
    --output terrain_box_check.png
```

### Step 5 – 3-D mesh plot

Two-panel 3-D figure showing terrain triangles and the detector box.

```bash
julia --project=. scripts/plot_terrain_3d.jl terrain.h5 box.ply \
    --r1 10000 \
    --r2 300 \
    --output terrain_3d.png
```

| Option | Description | Default |
|--------|-------------|---------|
| `--r1` | Outer panel horizontal radius (m) | `10000` |
| `--r2` | Inner panel horizontal radius (m) | `100` |
| `--output` | Output PNG file | `terrain_3d.png` |
| `--dpi` | Resolution (dots per inch) | `250` |

The box is automatically snapped to the terrain surface at the centre of the
patch using the closest terrain vertex.

---

## Complete example (copy-paste)

Running these commands in order reproduces the plots for the Jura site:

```bash
cd /path/to/CORSIKA_terrain

# Install packages (once only)
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Terrain mesh (~1 min for base_triangulation_100000)
julia --project=. scripts/create_terrain_geometry.jl 46.34249442 5.83882331 \
    --group base_triangulation_100000 \
    --output terrain.h5

# Convert to PLY
julia --project=. scripts/terrain_h5_to_ply.jl terrain.h5 --output terrain.ply

# Detector box (altitude derived from terrain mesh)
julia --project=. scripts/make_box_scene.jl \
    --lat 46.34249442 --lon 5.83882331 \
    --length 12.2 --width 2.44 --height 2.59 \
    --terrain-h5 terrain.h5 --yaw 90 \
    --output box.ply

# 2-D check
julia --project=. scripts/plot_terrain_box.jl terrain.h5 box.ply \
    --output terrain_box_check.png

# 3-D plot
julia --project=. scripts/plot_terrain_3d.jl terrain.h5 box.ply \
    --r2 300 --output terrain_3d.png
```

---

## Notes on mesh density

The `--group` argument controls the triangulation density.  The number in the
group name is approximate faces per steradian near the target.  Denser groups
produce finer meshes but take longer to process and require more memory.

| Group | Total vertices | Typical spacing at target | Useful `--max-angle` |
|-------|---------------|---------------------------|----------------------|
| `base_triangulation_1000`   |    3 000 | ~2 km   | 3–5° |
| `base_triangulation_10000`  |   30 000 | ~700 m  | 0.5–1° |
| `base_triangulation_100000` |  300 000 | ~55 m   | 0.05–0.1° |
| `base_triangulation_300000` |  900 000 | ~30 m   | 0.02–0.05° |

For the 3-D inner panel at 300 m radius, `base_triangulation_100000` with
`--max-angle 0.05` gives ~120 vertices in the inner panel (adequate for
visualisation).

---

## Environment variables reference

| Variable | Description |
|----------|-------------|
| `CORSIKA_TERRAIN_TRIANGULATION` | Path to `triangulation.h5` |
| `CORSIKA_TERRAIN_ELEVATION` | Path to `earth_relief_15s.h5` |
| `CORSIKA_TERRAIN_DATA` | Fallback directory if the above are unset |
| `CORSIKA_UPSTREAM_BUILD` | Path to `corsika_upstream` build directory (C++ app only) |
| `CORSIKA_DATA` | CORSIKA data directory (C++ app only) |
