#!/usr/bin/env python3
"""
3-D plot of shower particle hits, terrain mesh, and detector box.

Single 3-D axis in local ENU coordinates centred on the detector box,
showing all geometry within --radius metres of the origin.

Usage
-----
  python3 scripts/plot_shower_hits_3d.py TERRAIN.h5 BOX.ply PARTICLES.parquet
      [--output PATH] [--radius M] [--dpi N] [--elev DEG] [--azim DEG]
"""

import argparse
import struct

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

EARTH_RADIUS_M = 6.371e6

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def ecef_to_enu(ecef, lat0_deg, lon0_deg):
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    p0 = EARTH_RADIUS_M * np.array([
        np.cos(lat0)*np.cos(lon0),
        np.cos(lat0)*np.sin(lon0),
        np.sin(lat0)])
    east  = np.array([-np.sin(lon0),  np.cos(lon0),  0.0])
    north = np.array([-np.sin(lat0)*np.cos(lon0), -np.sin(lat0)*np.sin(lon0), np.cos(lat0)])
    up    = np.array([ np.cos(lat0)*np.cos(lon0),  np.cos(lat0)*np.sin(lon0), np.sin(lat0)])
    R  = np.vstack([east, north, up])
    dp = ecef - p0
    return (R @ dp.T).T

# ---------------------------------------------------------------------------
# PLY reader
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
    ap.add_argument('--output', default='shower_hits_3d.png')
    ap.add_argument('--radius', type=float, default=200.0,
                    help='Terrain face centroid radius for display (m)')
    ap.add_argument('--xlim', type=float, default=50.0,
                    help='Half-width of plot axes (m)')
    ap.add_argument('--dpi',  type=int,   default=200)
    ap.add_argument('--elev', type=float, default=25.0,
                    help='Camera elevation angle (deg)')
    ap.add_argument('--azim', type=float, default=-60.0,
                    help='Camera azimuth angle (deg)')
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
        v_raw = g['vertices'][:]
        fa    = g['faces'][:]
        lat0  = float(g.attrs['lat_deg'])
        lon0  = float(g.attrs['lon_deg'])

    terrain_ecef  = v_raw.T if v_raw.shape[0] == 3 else v_raw
    terrain_faces = fa.T    if fa.shape[0]     == 3 else fa
    print(f'  {terrain_ecef.shape[0]} vertices, {terrain_faces.shape[0]} faces')

    # ── Load box ─────────────────────────────────────────────────────────────
    print(f'Reading box: {args.box_ply}')
    box_ecef, box_faces = read_ply(args.box_ply)
    print(f'  {len(box_ecef)} vertices, {len(box_faces)} faces')

    # ── Load particles ────────────────────────────────────────────────────────
    print(f'Reading particles: {args.particles_parquet}')
    tbl    = pq.read_table(args.particles_parquet).to_pydict()
    px_loc = np.array(tbl['x'], dtype=float)   # CORSIKA local x ≈ North
    py_loc = np.array(tbl['y'], dtype=float)   # CORSIKA local y ≈ East
    pz_loc = np.array(tbl['z'], dtype=float)   # CORSIKA local z ≈ Up (relative)
    ke     = np.array(tbl['kinetic_energy'], dtype=float)
    pdg    = np.array(tbl['pdg'], dtype=int)
    n_part = len(px_loc)
    print(f'  {n_part} particle hits')

    # ── Project terrain & box to ENU ─────────────────────────────────────────
    terrain_enu = ecef_to_enu(terrain_ecef, lat0, lon0)
    box_enu     = ecef_to_enu(box_ecef,     lat0, lon0)
    box_cen     = box_enu.mean(axis=0)
    print(f'  Box centroid ENU: East={box_cen[0]:.1f} m, '
          f'North={box_cen[1]:.1f} m, Alt={box_cen[2]:.1f} m')

    # Particle positions are ECEF offsets from the box centroid.
    # Recover ENU via the site rotation matrix.
    lat0_r = np.radians(lat0); lon0_r = np.radians(lon0)
    R_enu = np.vstack([
        np.array([-np.sin(lon0_r),  np.cos(lon0_r),  0.0]),
        np.array([-np.sin(lat0_r)*np.cos(lon0_r),
                   -np.sin(lat0_r)*np.sin(lon0_r), np.cos(lat0_r)]),
        np.array([ np.cos(lat0_r)*np.cos(lon0_r),
                    np.cos(lat0_r)*np.sin(lon0_r), np.sin(lat0_r)]),
    ])
    xyz_offset = np.column_stack([px_loc, py_loc, pz_loc])
    enu_off = (R_enu @ xyz_offset.T).T          # (N,3) ENU offsets from box centroid
    part_east  = enu_off[:, 0] + box_cen[0]
    part_north = enu_off[:, 1] + box_cen[1]
    part_up    = enu_off[:, 2] + box_cen[2]

    # ── Filter terrain faces within radius (centroid-based) ──────────────────
    face_cen_e = terrain_enu[terrain_faces, 0].mean(axis=1)
    face_cen_n = terrain_enu[terrain_faces, 1].mean(axis=1)
    nearby = np.where(np.hypot(face_cen_e, face_cen_n) <= args.radius)[0]
    print(f'  {len(nearby)} terrain faces within {args.radius:.0f} m (centroid)')

    vis_idx = np.unique(terrain_faces[nearby].ravel())
    alt_lo  = terrain_enu[vis_idx, 2].min() if len(vis_idx) else box_cen[2] - 10
    alt_hi  = terrain_enu[vis_idx, 2].max() if len(vis_idx) else box_cen[2] + 10

    # ── Colour scale for kinetic energy ──────────────────────────────────────
    log_ke = np.log10(np.clip(ke, 1e-4, None))
    ke_lo, ke_hi = log_ke.min(), log_ke.max()

    # PDG → marker / label
    markers = {22: 'o', 11: 's', -11: 's', 13: '^', -13: '^'}
    labels  = {22: 'photon', 11: 'e\u207b', -11: 'e\u207a', 13: '\u03bc\u207b', -13: '\u03bc\u207a'}

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 8))
    ax  = fig.add_subplot(111, projection='3d')

    n_str = f'{n_part} hits' if n_part else 'no hits'
    ax.set_title(
        f'3-D shower hits  lat={lat0:.3f}\u00b0 lon={lon0:.3f}\u00b0  ({n_str})',
        fontsize=12)
    ax.set_xlabel('East  (m)',     labelpad=8)
    ax.set_ylabel('North  (m)',    labelpad=8)
    ax.set_zlabel('Altitude  (m)', labelpad=8)

    # ── Terrain mesh (plot_trisurf for per-vertex altitude colouring) ────────
    if len(nearby):
        vis_idx   = np.unique(terrain_faces[nearby].ravel())
        remap     = np.empty(terrain_enu.shape[0], dtype=int)
        remap[vis_idx] = np.arange(len(vis_idx))
        local_faces = remap[terrain_faces[nearby]]        # reindexed (M,3)

        tx = terrain_enu[vis_idx, 0]
        ty = terrain_enu[vis_idx, 1]
        tz = terrain_enu[vis_idx, 2]

        ts = ax.plot_trisurf(tx, ty, tz, triangles=local_faces,
                             cmap='terrain', vmin=alt_lo, vmax=alt_hi,
                             alpha=0.70, edgecolor=(0, 0, 0, 0.3), linewidth=0.3)
        fig.colorbar(ts, ax=ax, label='Altitude  (m)',
                     shrink=0.55, pad=0.1, fraction=0.03)

    # ── Box edges ────────────────────────────────────────────────────────────
    for face in box_faces:
        idx = list(face) + [face[0]]
        ax.plot(box_enu[idx, 0], box_enu[idx, 1], box_enu[idx, 2],
                color='red', linewidth=0.7, alpha=0.7)

    # ── Particle hits ─────────────────────────────────────────────────────────
    sc = None
    if n_part > 0:
        for code in np.unique(pdg):
            mask = pdg == code
            sc = ax.scatter(part_east[mask], part_north[mask], part_up[mask],
                            c=log_ke[mask], cmap='plasma', vmin=ke_lo, vmax=ke_hi,
                            s=60, marker=markers.get(code, 'o'),
                            edgecolors='k', linewidths=0.4, zorder=5,
                            depthshade=False,
                            label=labels.get(code, str(code)))
        ax.legend(fontsize=9, loc='upper left', framealpha=0.7)
        cb2 = fig.colorbar(sc, ax=ax, label='log\u2081\u2080(KE / GeV)',
                           shrink=0.55, pad=0.15, fraction=0.03)

    # ── Axis limits ──────────────────────────────────────────────────────────
    r = args.xlim
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    # z: tight around terrain + a few metres above box top
    box_alt_hi = box_enu[:, 2].max()
    z_lo = alt_lo - 2
    z_hi = max(alt_hi, box_alt_hi) + 5
    ax.set_zlim(z_lo, z_hi)

    ax.view_init(elev=args.elev, azim=args.azim)

    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
