"""Generate synthetic inputs and a numpy reference for the CUDA ray_trace_1 test.

Builds a small axis-aligned grid of cells, a handful of random star positions, and
uses every cell center as a ray target (matching Halo, which derives targets from
cell centers). Runs a numpy port of RadiativeTransfer.ray_trace_1 (cell_based=True)
as the oracle and writes both the inputs the Halo loader expects and the reference
outputs the C++ test compares against.

Usage:
    python gen_rt1_reference.py <out_dir> [halo_name]

Writes into <out_dir>/<halo_name>/:
    grid_ll.npy, grid_ur.npy, stars_positions.npy    (inputs read by Halo::init)
    ref_dr.npy, ref_target.npy, ref_cell.npy, ref_star.npy   (oracle outputs)
"""
import os
import sys
import numpy as np


def ray_trace_1_oracle(ll, ur, ipos, fpos, cell_based=True):
    """Port of RadiativeTransfer.ray_trace_1 (dims=3). Loops over cells to keep memory small.

    Returns (dr, target_ind, cell_ind, star_ind) where target indexes fpos, star indexes
    ipos, and cell indexes ll/ur. Everything in float64.
    """
    n_cells = ll.shape[0]
    # M[target, star] = fpos[target] - ipos[star]
    M = fpos[:, None, :] - ipos[None, :, :]  # [n_fpos, n_ipos, 3]

    tgt_list, cell_list, star_list, dr_list = [], [], [], []
    with np.errstate(divide="ignore", invalid="ignore"):
        for c in range(n_cells):
            t0 = (ll[c] - ipos[None, :, :]) / M  # [n_fpos, n_ipos, 3]
            t1 = (ur[c] - ipos[None, :, :]) / M
            tmin = np.minimum(t0, t1).max(axis=2)  # [n_fpos, n_ipos]
            tmax = np.maximum(t0, t1).min(axis=2)
            hit = (tmin <= tmax) & (tmin <= 1.0) & (tmax >= 0.0)
            if not hit.any():
                continue
            ti, si = np.where(hit)  # target (fpos) idx, star (ipos) idx
            tmin_f = np.maximum(tmin[hit], 0.0)
            tmax_f = tmax[hit] if cell_based else np.minimum(tmax[hit], 1.0)
            Mh = M[ti, si]  # [K, 3]
            span = (tmax_f - tmin_f)[:, None]
            drc = np.linalg.norm(span * Mh, axis=1)
            tgt_list.append(ti)
            star_list.append(si)
            cell_list.append(np.full(ti.shape, c))
            dr_list.append(drc)

    if not dr_list:
        return (np.empty(0), np.empty(0, int), np.empty(0, int), np.empty(0, int))
    return (
        np.concatenate(dr_list),
        np.concatenate(tgt_list),
        np.concatenate(cell_list),
        np.concatenate(star_list),
    )


def main():
    if len(sys.argv) < 2:
        print("usage: gen_rt1_reference.py <out_dir> [halo_name]", file=sys.stderr)
        sys.exit(1)
    out_dir = sys.argv[1]
    halo_name = sys.argv[2] if len(sys.argv) > 2 else "synth"
    d = os.path.join(out_dir, halo_name)
    os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(12345)

    # --- synthetic grid: n_side^3 axis-aligned cells at CGS scale ---
    n_side = 6
    L = 3.0e20  # cell edge length [cm]
    origin = np.array([1.0e23, 2.0e23, -5.0e22], dtype=np.float64)  # CGS-scale offset

    idx = np.arange(n_side)
    I, J, K = np.meshgrid(idx, idx, idx, indexing="ij")
    ijk = np.stack([I.ravel(), J.ravel(), K.ravel()], axis=1).astype(np.float64)
    ll = origin + ijk * L
    ur = ll + L
    centers = 0.5 * (ll + ur)
    n_cells = ll.shape[0]

    # --- stars: random points in an expanded box so rays cross many cells ---
    n_stars = 12
    lo = origin - L
    hi = origin + n_side * L + L
    spos = rng.uniform(lo, hi, size=(n_stars, 3)).astype(np.float64)

    # targets = every cell center (matches Halo::computeCenters)
    fpos = centers

    dr, tgt, cel, sta = ray_trace_1_oracle(ll, ur, spos, fpos, cell_based=True)

    # inputs consumed by Halo::init
    np.save(os.path.join(d, "grid_ll.npy"), ll.astype(np.float64))
    np.save(os.path.join(d, "grid_ur.npy"), ur.astype(np.float64))
    np.save(os.path.join(d, "stars_positions.npy"), spos.astype(np.float64))

    # oracle outputs consumed by test_rt1
    np.save(os.path.join(d, "ref_dr.npy"), dr.astype(np.float64))
    np.save(os.path.join(d, "ref_target.npy"), tgt.astype(np.uint32))
    np.save(os.path.join(d, "ref_cell.npy"), cel.astype(np.uint32))
    np.save(os.path.join(d, "ref_star.npy"), sta.astype(np.uint32))

    print(
        "wrote %s: %d cells, %d stars, %d targets -> %d intersections"
        % (d, n_cells, n_stars, fpos.shape[0], dr.shape[0])
    )


if __name__ == "__main__":
    main()
