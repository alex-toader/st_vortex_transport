"""
parallel_fdtd.py — Parallel FDTD helper for w_21_kappa scripts.

Uses fork-based multiprocessing: workers inherit all parent state
(gamma_pml, sphere points, etc.) via copy-on-write. No pickle needed.

n_workers=1 → serial (identical to existing scripts).
n_workers>1 → Pool(n_workers) with fork.

Benchmark (M3, L=80): fork(2) → 1.95×, fork(4) → 2.48×.

Usage:
    from parallel_fdtd import compute_references, compute_scattering

    refs = compute_references(k_vals, ..., n_workers=4)
    sigma_tr = compute_scattering(k_vals, refs, alpha, R, ..., n_workers=4)
"""

import multiprocessing as mp
import numpy as np

from elastic_3d import scalar_laplacian_3d, make_damping_3d, run_fdtd_3d
from gauge_3d import make_vortex_force
from scattering_3d import (make_wave_packet_3d, make_sphere_points,
                           compute_sphere_f2, integrate_sigma_3d,
                           estimate_n_steps_3d)

# Fork: workers inherit parent memory (gamma, sphere pts) via COW.
# Must be set before any Pool is created.
try:
    mp.set_start_method('fork')
except RuntimeError:
    pass  # already set


# Module-level state, set by _init_* before Pool.map.
# Workers inherit these via fork — no pickle.
# NOT thread-safe: concurrent calls with different parameters will collide.
# This is a serial-workflow helper (one compute_references + N compute_scattering).
_gamma_pml = None
_iz = _iy = _ix = None
_K1 = _K2 = _DT = None
_x_start = _sx = _r_m = None
_L = None
_thetas = _phis = None


def _init_shared(L, DW, DS, DT, sx, r_m, thetas, phis, K1, K2):
    """Set module-level shared state before forking workers."""
    global _gamma_pml, _iz, _iy, _ix, _K1, _K2, _DT
    global _x_start, _sx, _r_m, _L, _thetas, _phis
    _gamma_pml = make_damping_3d(L, DW, DS)
    _iz, _iy, _ix = make_sphere_points(L, r_m, thetas, phis)
    _K1, _K2, _DT = K1, K2, DT
    _x_start, _sx, _r_m = DW + 5, sx, r_m
    _L = L
    _thetas, _phis = thetas, phis


def _run_ref(k0):
    """Single reference FDTD run. Uses module-level shared state."""
    ux0, vx0 = make_wave_packet_3d(_L, k0, _x_start, _sx, _K1, _K2)
    ns = estimate_n_steps_3d(k0, _L, _x_start, _sx, _r_m, _DT, 50, _K1, _K2)
    def force_plain(ux, uy, uz):
        return scalar_laplacian_3d(ux, uy, uz, _K1, _K2)
    ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), _gamma_pml,
                      _DT, ns, rec_iz=_iz, rec_iy=_iy, rec_ix=_ix, rec_n=ns)
    return (k0, ref, ux0, vx0, ns)


# Scattering worker state (set per R value)
_f_def = None
_refs = None


def _init_scatter(refs, alpha, R, L, K1, K2):
    """Set scattering-specific state before forking workers."""
    global _f_def, _refs
    assert _gamma_pml is not None, \
        "call compute_references before compute_scattering"
    assert _L == L, \
        f"L mismatch: shared state has L={_L}, caller passed L={L}"
    _f_def = make_vortex_force(alpha, R, L, K1, K2)
    _refs = refs


def _run_scatter(k0):
    """Single scattering FDTD + sigma_tr. Uses module-level shared state."""
    ref, ux0, vx0, ns = _refs[k0]
    d = run_fdtd_3d(_f_def, ux0.copy(), vx0.copy(), _gamma_pml,
                    _DT, ns, rec_iz=_iz, rec_iy=_iy, rec_ix=_ix, rec_n=ns)
    f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                           ref['ux'], ref['uy'], ref['uz'], _r_m)
    _, st = integrate_sigma_3d(f2, _thetas, _phis)
    return (k0, st)


def compute_references(k_vals, L, DW, DS, DT, sx, r_m,
                       thetas, phis, K1, K2, n_workers=1):
    """Compute free-field references for all k values.

    Returns: dict {k0: (ref_dict, ux0, vx0, ns)}.
    """
    _init_shared(L, DW, DS, DT, sx, r_m, thetas, phis, K1, K2)

    if n_workers <= 1:
        refs = {}
        for k0 in k_vals:
            _, ref, ux0, vx0, ns = _run_ref(k0)
            refs[k0] = (ref, ux0, vx0, ns)
        return refs

    with mp.Pool(n_workers) as p:
        results = p.map(_run_ref, list(k_vals))
    return {k0: (ref, ux0, vx0, ns) for k0, ref, ux0, vx0, ns in results}


def compute_scattering(k_vals, refs, alpha, R, L, DW, DS, DT,
                       r_m, thetas, phis, K1, K2, n_workers=1):
    """Scattering FDTD + sigma_tr for all k, given references.

    Returns: np.array of sigma_tr values, one per k.
    """
    _init_scatter(refs, alpha, R, L, K1, K2)

    if n_workers <= 1:
        sigma_tr = np.zeros(len(k_vals))
        for i, k0 in enumerate(k_vals):
            _, st = _run_scatter(k0)
            sigma_tr[i] = st
        return sigma_tr

    with mp.Pool(n_workers) as p:
        results = p.map(_run_scatter, list(k_vals))

    result_dict = {k0: st for k0, st in results}
    return np.array([result_dict[k0] for k0 in k_vals])
