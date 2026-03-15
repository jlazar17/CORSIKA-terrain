#!/usr/bin/env python3
"""
Plot shower particle hits on the detector box and terrain mesh.

Two-panel figure in local ENU coordinates:
  Left  – 300 m top-down view: terrain triangles coloured by altitude,
           box footprint (red), particle hits coloured by log10(KE/GeV).
  Right – Zoom ±15 m around box: box triangle edges (red),
           particle hits coloured by log10(KE/GeV).

Usage
-----
  python3 scripts/plot_shower_hits.py TERRAIN.h5 BOX.ply PARTICLES.parquet
      [--output PATH] [--radius M] [--zoom M] [--dpi N]
"""

import argparse
import struct
import sys

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import pyarrow.parquet as pq

EARTH_RADIUS_M = 6.371e6

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def ecef_to_enu(ecef, lat0_deg, lon0_deg):
    """Convert (N,3) ECEF array to local ENU centred on (lat0, lon0), altitude=0."""
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    p0 = EARTH_RADIUS_M * np.array([
        np.cos(lat0)*np.cos(lon0),
        np.cos(lat0)*np.sin(lon0),
        np.sin(lat0)])
    east  = np.array([-np.sin(lon0),  np.cos(lon0),  0.0])
    north = np.array([-np.sin(lat0)*np.cos(lon0), -np.sin(lat0)*np.sin(lon0), np.cos(lat0)])
    up    = np.array([ np.cos(lat0)*np.cos(lon0),  np.cos(lat0)*np.sin(lon0), np.sin(lat0)])
    R  = np.vstack([east, north, up])   # (3, 3)
    dp = ecef - p0                       # (N, 3)
    return (R @ dp.T).T                  # (N, 3) -> columns: East, North, Up

# ---------------------------------------------------------------------------
# PLY reader (binary_little_endian, double or float vertices)
# ---------------------------------------------------------------------------

def _ply_fmt(t):
    return {'float': ('f', 4), 'float32': ('f', 4),
            'double': ('d', 8), 'float64': ('d', 8),
            'int': ('i', 4), 'int32': ('i', 4),
            'uint': ('I', 4), 'uint32': ('I', 4),
            'uchar': ('B', 1), 'uint8': ('B', 1)}[t]

def read_ply(path):
    with open(path, 'rb') as f:
        n_verts = n_faces = 0
        vprops = []
        fcount_fmt = ('B', 1)
        fidx_fmt   = ('I', 4)
        current = None
        while True:
            line = f.readline().decode('ascii').strip()
            if line == 'end_header':
                break
            tok = line.split()
            if not tok:
                continue
            if tok[0] == 'element':
                current = tok[1]
                if tok[1] == 'vertex': n_verts = int(tok[2])
                if tok[1] == 'face':   n_faces = int(tok[2])
            elif tok[0] == 'property' and current == 'vertex' and tok[1] != 'list':
                vprops.append((tok[-1], tok[1]))
            elif tok[0] == 'property' and current == 'face' and tok[1] == 'list':
                fcount_fmt = _ply_fmt(tok[2])
                fidx_fmt   = _ply_fmt(tok[3])

        xi = next(i for i,(n,_) in enumerate(vprops) if n=='x')
        yi = next(i for i,(n,_) in enumerate(vprops) if n=='y')
        zi = next(i for i,(n,_) in enumerate(vprops) if n=='z')
        fmts = [_ply_fmt(t) for _,t in vprops]
        row_size = sum(s for _,s in fmts)

        verts = np.zeros((n_verts, 3))
        for i in range(n_verts):
            row = f.read(row_size)
            off = 0
            vals = []
            for fmt, size in fmts:
                vals.append(struct.unpack_from('<'+fmt, row, off)[0])
                off += size
            verts[i] = [vals[xi], vals[yi], vals[zi]]

        faces = []
        for _ in range(n_faces):
            cnt = struct.unpack('<'+fcount_fmt[0], f.read(fcount_fmt[1]))[0]
            idx = [struct.unpack('<'+fidx_fmt[0], f.read(fidx_fmt[1]))[0]
                   for _ in range(cnt)]
            faces.append(idx)

    return verts, faces

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('terrain_h5')
    ap.add_argument('box_ply')
    ap.add_argument('particles_parquet')
    ap.add_argument('--output', default='shower_hits.png')
    ap.add_argument('--radius', type=float, default=300.0,
                    help='Terrain display radius in left panel (m)')
    ap.add_argument('--zoom', type=float, default=15.0,
                    help='Half-width of zoom panel (m)')
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('--group', default='',
                    help='HDF5 group for terrain vertices/faces')
    args = ap.parse_args()

    # ── Load terrain ─────────────────────────────────────────────────────────
    print(f'Reading terrain: {args.terrain_h5}')
    with h5py.File(args.terrain_h5, 'r') as f:
        gname = args.group or next(
            k for k in f if 'vertices' in f[k] and 'faces' in f[k])
        print(f'  Group: {gname}')
        g = f[gname]
        # HDF5.jl stores (N,3) Julia arrays as (3,N) on disk; h5py reads (3,N)
        v_raw = g['vertices'][:]   # h5py shape: (3, N)
        fa    = g['faces'][:]      # h5py shape: (3, M), 0-based
        lat0  = float(g.attrs['lat_deg'])
        lon0  = float(g.attrs['lon_deg'])

    # h5py returns (3, N) -> transpose to (N, 3)
    terrain_ecef = v_raw.T if v_raw.shape[0] == 3 else v_raw
    terrain_faces = fa.T  if fa.shape[0]     == 3 else fa  # (M, 3)
    print(f'  {terrain_ecef.shape[0]} vertices, {terrain_faces.shape[0]} faces')

    # ── Load box ─────────────────────────────────────────────────────────────
    print(f'Reading box: {args.box_ply}')
    box_ecef, box_faces = read_ply(args.box_ply)
    print(f'  {len(box_ecef)} vertices, {len(box_faces)} faces')

    # ── Load particles ────────────────────────────────────────────────────────
    print(f'Reading particles: {args.particles_parquet}')
    tbl = pq.read_table(args.particles_parquet).to_pydict()
    px = np.array(tbl['x'], dtype=float)   # CORSIKA local x (≈ North, metres)
    py = np.array(tbl['y'], dtype=float)   # CORSIKA local y (≈ East,  metres)
    ke = np.array(tbl['kinetic_energy'], dtype=float)   # GeV
    pdg = np.array(tbl['pdg'], dtype=int)
    n_part = len(px)
    print(f'  {n_part} particle hits')
    pdg_labels = {22: 'γ', 11: 'e⁻', -11: 'e⁺', 13: 'μ⁻', -13: 'μ⁺'}
    for code, count in zip(*np.unique(pdg, return_counts=True)):
        print(f'    PDG {code:5d} ({pdg_labels.get(code,"?")}): {count}')

    # ── Project terrain & box to ENU ─────────────────────────────────────────
    terrain_enu = ecef_to_enu(terrain_ecef, lat0, lon0)  # (N,3): East, North, Up
    box_enu     = ecef_to_enu(box_ecef,     lat0, lon0)

    # Box centroid in ENU (should be ≈ (0, 0, altitude))
    box_cen = box_enu.mean(axis=0)
    print(f'  Box centroid ENU: East={box_cen[0]:.1f} m, '
          f'North={box_cen[1]:.1f} m, Alt={box_cen[2]:.1f} m')

    # Particle positions are ECEF offsets from the box centroid.
    # Convert to local ENU via the site rotation matrix.
    lat0_r = np.radians(lat0); lon0_r = np.radians(lon0)
    east_hat  = np.array([-np.sin(lon0_r),  np.cos(lon0_r),  0.0])
    north_hat = np.array([-np.sin(lat0_r)*np.cos(lon0_r),
                           -np.sin(lat0_r)*np.sin(lon0_r), np.cos(lat0_r)])
    up_hat    = np.array([ np.cos(lat0_r)*np.cos(lon0_r),
                            np.cos(lat0_r)*np.sin(lon0_r), np.sin(lat0_r)])
    R_enu = np.vstack([east_hat, north_hat, up_hat])   # (3,3)

    pz = np.array(tbl['z'], dtype=float)
    xyz_ecef_offset = np.column_stack([px, py, pz])    # (N,3) ECEF offsets
    enu_offsets = (R_enu @ xyz_ecef_offset.T).T         # (N,3) ENU offsets

    part_east  = enu_offsets[:, 0] + box_cen[0]
    part_north = enu_offsets[:, 1] + box_cen[1]

    # ── Filter terrain faces within radius ───────────────────────────────────
    face_cen_e = terrain_enu[terrain_faces, 0].mean(axis=1)
    face_cen_n = terrain_enu[terrain_faces, 1].mean(axis=1)
    dist2d = np.hypot(face_cen_e, face_cen_n)
    nearby = np.where(dist2d <= args.radius)[0]
    print(f'  {len(nearby)} terrain faces within {args.radius:.0f} m')

    # Altitude colour limits for visible faces.
    vis_idx = np.unique(terrain_faces[nearby].ravel())
    alt_lo  = terrain_enu[vis_idx, 2].min() if len(vis_idx) else 0
    alt_hi  = terrain_enu[vis_idx, 2].max() if len(vis_idx) else 1000

    # ── Colour scale for kinetic energy ──────────────────────────────────────
    log_ke = np.log10(np.clip(ke, 1e-4, None))
    ke_lo, ke_hi = log_ke.min(), log_ke.max()

    # PDG → marker
    markers = {22: 'o', 11: 's', -11: 's', 13: '^', -13: '^'}
    labels  = {22: 'photon', 11: 'e⁻', -11: 'e⁺', 13: 'μ⁻', -13: 'μ⁺'}

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle(
        f'Shower hits — lat={lat0:.3f}° lon={lon0:.3f}°  ({n_part} hits)',
        fontsize=13)

    # ─ Panel 1: 300 m top-down ────────────────────────────────────────────────
    ax1.set_title(f'Top-down  {args.radius:.0f} m view')
    ax1.set_xlabel('East  (m)')
    ax1.set_ylabel('North  (m)')
    ax1.set_aspect('equal')

    # Terrain triangles coloured by altitude.
    if len(nearby):
        triang = mtri.Triangulation(
            terrain_enu[:, 0], terrain_enu[:, 1],
            triangles=terrain_faces[nearby])
        tc = ax1.tripcolor(triang, terrain_enu[:, 2],
                           cmap='terrain', vmin=alt_lo, vmax=alt_hi,
                           shading='flat', alpha=0.8, edgecolors='k', linewidth=0.3)
        fig.colorbar(tc, ax=ax1, label='Altitude  (m)', fraction=0.046, pad=0.04)

    # Box footprint (ENU East vs North).
    e_lo, e_hi = box_enu[:, 0].min(), box_enu[:, 0].max()
    n_lo, n_hi = box_enu[:, 1].min(), box_enu[:, 1].max()
    rect = plt.Polygon([[e_lo,n_lo],[e_hi,n_lo],[e_hi,n_hi],[e_lo,n_hi]],
                       closed=True, fill=False, edgecolor='red', linewidth=2, zorder=3)
    ax1.add_patch(rect)

    # Particle hits.
    cmap_ke = plt.cm.plasma
    if n_part > 0:
        sc = ax1.scatter(part_east, part_north, c=log_ke,
                         cmap='plasma', vmin=ke_lo, vmax=ke_hi,
                         s=40, zorder=4, edgecolors='k', linewidths=0.5)

    ax1.set_xlim(-args.radius, args.radius)
    ax1.set_ylim(-args.radius, args.radius)

    # ─ Panel 2: box zoom ──────────────────────────────────────────────────────
    z = args.zoom
    ax2.set_title(f'Box zoom  ±{z:.0f} m')
    ax2.set_xlabel('East  (m)')
    ax2.set_ylabel('North  (m)')
    ax2.set_aspect('equal')

    # Box triangle edges in ENU (East vs North).
    for face in box_faces:
        idx  = list(face) + [face[0]]
        xs   = box_enu[idx, 0]
        ys   = box_enu[idx, 1]
        ax2.plot(xs, ys, color='red', linewidth=0.8, alpha=0.6)

    # Particle hits.
    if n_part > 0:
        for code in np.unique(pdg):
            mask = pdg == code
            sc2 = ax2.scatter(part_east[mask], part_north[mask],
                              c=log_ke[mask], cmap='plasma',
                              vmin=ke_lo, vmax=ke_hi,
                              s=60, marker=markers.get(code, 'o'),
                              edgecolors='k', linewidths=0.5, zorder=4,
                              label=labels.get(code, str(code)))
        ax2.legend(fontsize=8, loc='upper right')
        cb = fig.colorbar(sc2, ax=ax2, label='log₁₀(KE / GeV)',
                          fraction=0.046, pad=0.04)

    ax2.set_xlim(-z, z)
    ax2.set_ylim(-z, z)

    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f'Saved: {args.output}')

if __name__ == '__main__':
    main()
