"""Convergence scan: sigma_bond + sigma_ring at L=80, 100, 120.

Tests whether L=100 is sufficient for all FDTD datasets.
Compares with hardcoded data in data/.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 convergence_scan.py
Actual time: 44 min (M3 MacBook Air, Mar 2026).

Results:
  sigma_bond (single z-bond, α=0.30):
    k      L= 80    L=100    L=120    Δ80/100  Δ100/120
    0.3      0.060    0.047    0.047     27.5%      0.3%
    0.5      0.054    0.049    0.048     11.9%      1.4%
    0.7      0.057    0.052    0.052     10.3%      0.6%
    0.9      0.062    0.056    0.056     11.0%      0.3%
    1.1      0.071    0.063    0.064     12.6%      1.4%
    1.3      0.081    0.072    0.072     11.7%      0.2%
    1.5      0.096    0.086    0.086     12.3%      0.5%
    Max Δ(100 vs 120) = 1.4%  CONVERGED

  sigma_ring R=5 (81 bonds, α=0.30):
    k      L= 80    L=100    L=120    Δ80/100  Δ100/120
    0.3     40.743   28.331   29.105     43.8%      2.7%
    0.5     14.158   12.311   12.392     15.0%      0.7%
    0.7      7.694    7.090    7.128      8.5%      0.5%
    0.9      5.048    4.688    4.698      7.7%      0.2%
    1.1      3.776    3.495    3.519      8.0%      0.7%
    1.3      3.138    2.912    2.915      7.7%      0.1%
    1.5      2.806    2.601    2.611      7.9%      0.4%
    Max Δ(100 vs 120) = 2.7%  CONVERGED

  sigma_ring R=9 (225 bonds, α=0.30):
    k      L= 80    L=100    L=120    Δ80/100  Δ100/120
    0.3    117.843   78.060   80.676     51.0%      3.2%
    0.5     34.542   31.301   31.202     10.4%      0.3%
    0.7     19.429   18.018   18.145      7.8%      0.7%
    0.9     13.340   12.377   12.413      7.8%      0.3%
    1.1     10.289    9.561    9.578      7.6%      0.2%
    1.3      8.563    7.973    7.984      7.4%      0.1%
    1.5      7.745    7.215    7.250      7.3%      0.5%
    Max Δ(100 vs 120) = 3.2%  CONVERGED

  Conclusion: L=100 converged for all cases. Production L = 100.

  NOTE — two distinct bias mechanisms at L=80:
  1. PML contamination at low k (long λ): affects sigma_ring heavily at k=0.3
     (44-51%), drops to 7-10% at k≥0.5. Clear mechanism: measurement sphere
     too close to PML at long wavelength.
  2. Systematic ~12% bias in sigma_bond at ALL k (10-13% at k≥0.5, 27% at
     k=0.3 = PML + this effect). Cause unclear — possibly near-field or
     insufficient box for single-scatterer measurement. Not PML alone,
     because it persists at high k where PML bias vanishes for ring.
  Both effects converge away at L=100 (max 1.4% vs L=120 for bond,
  max 3.2% for ring R=9).
"""
import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from elastic_3d import scalar_laplacian_3d, make_damping_3d, run_fdtd_3d
from gauge_3d import make_vortex_force
from scattering_3d import (make_wave_packet_3d, make_sphere_points,
                           compute_sphere_f2, integrate_sigma_3d,
                           estimate_n_steps_3d)

# ── Parameters ────────────────────────────────────────────────────
K1, K2 = 1.0, 0.5
ALPHA = 0.30
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
THETAS = np.linspace(0, np.pi, 13)
PHIS = np.linspace(0, 2 * np.pi, 24, endpoint=False)

k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])

L_values = [80, 100, 120]


# ── FDTD runner ──────────────────────────────────────────────────

def run_sigma_tr(L, R_loop, k_arr):
    """Run FDTD at box size L with vortex ring of radius R_loop."""
    center = L // 2
    x_start = center - 20

    gamma_pml = make_damping_3d(L, DW, DS)
    iz, iy, ix = make_sphere_points(L, r_m, THETAS, PHIS)

    def force_plain(ux, uy, uz):
        return scalar_laplacian_3d(ux, uy, uz, K1, K2)

    f_def = make_vortex_force(ALPHA, R_loop, L, K1, K2)

    sigma_tr = np.zeros(len(k_arr))
    for i, k0 in enumerate(k_arr):
        ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
        ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)

        ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma_pml,
                          DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        d = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma_pml,
                        DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                               ref['ux'], ref['uy'], ref['uz'], r_m)
        _, st = integrate_sigma_3d(f2, THETAS, PHIS)
        sigma_tr[i] = st
        print(f"      k={k0:.1f}: σ_tr = {st:.3f}  (ns={ns})", flush=True)

    return sigma_tr


def print_comparison(label, results, stored=None):
    """Print convergence table for one dataset."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # Header
    header = "  k    "
    for L in L_values:
        header += f"  L={L:3d}  "
    header += "  Δ80/100  Δ100/120"
    if stored is not None:
        header += "  stored   Δstored/100"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for ik, k in enumerate(k_vals):
        row = f"  {k:.1f}  "
        vals = [results[L][ik] for L in L_values]
        for v in vals:
            row += f"  {v:7.3f}"

        # Deltas
        d80_100 = abs(vals[0] - vals[1]) / vals[1] * 100
        d100_120 = abs(vals[1] - vals[2]) / vals[2] * 100
        row += f"    {d80_100:5.1f}%    {d100_120:5.1f}%"

        if stored is not None:
            s = stored[ik]
            d_s_100 = abs(s - vals[1]) / vals[1] * 100
            row += f"   {s:7.3f}     {d_s_100:5.1f}%"

        print(row)

    # Summary
    d100_120_all = [abs(results[100][ik] - results[120][ik]) / results[120][ik] * 100
                    for ik in range(len(k_vals))]
    print(f"\n  Max Δ(100 vs 120) = {max(d100_120_all):.1f}%  "
          f"{'CONVERGED' if max(d100_120_all) < 5 else 'NOT CONVERGED'}")


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.sigma_bond import sigma_bond
    from data.sigma_ring import sigma_ring

    cases = [
        ("sigma_bond (single z-bond, α=0.30)", 0, sigma_bond),
        ("sigma_ring R=5 (81 bonds, α=0.30)", 5, sigma_ring[5]),
        ("sigma_ring R=9 (225 bonds, α=0.30)", 9, sigma_ring[9]),
    ]

    t_total = time.time()

    for label, R_loop, stored in cases:
        print(f"\n>>> Running: {label}")
        results = {}
        for L in L_values:
            t0 = time.time()
            print(f"    L={L} ...", end=" ", flush=True)
            results[L] = run_sigma_tr(L, R_loop, k_vals)
            dt = time.time() - t0
            print(f"done ({dt:.0f}s)")

        print_comparison(label, results, stored)

    print(f"\n\nTotal time: {(time.time() - t_total)/60:.1f} min")
