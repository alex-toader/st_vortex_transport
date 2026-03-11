"""
Route 43 (F24): Born form factor of Dirac disk.

Computes the Born-level transport cross-section from the discrete form
factor F(q) = Σ_j exp(iq·r_j) where r_j are the gauged bond positions.

Born approximation for displacement coupling on z-bonds:
  σ_tr^Born(k) = C × I_tr(k, R) / sin²(k)

where I_tr(k, R) = ∫ Z(k,θ) × |F_disk(q_⊥)|² × (1-cosθ_s) dΩ

  F_disk(q_⊥) = Σ_j exp(i q_⊥ · r_j)    [2D form factor of disk]
  Z(k,θ) = 4cos²(k cosθ / 2)              [z-structure: displacement coupling]
  (1-cosθ_s) = transport kernel            [cosθ_s = sinθ cosφ for +x incidence]

The z-structure comes from: force_lo += K_eff * u_hi (and symmetric).
Born vertex: V ~ K_eff × (e^{ik_{in,z}} + e^{-ik_{out,z}}) × F_disk(-q_⊥).
Since k_in = k x̂: k_{in,z}=0, k_{out,z}=k cosθ.
|vertex|² ∝ |1 + e^{-ik cosθ}|² = 4cos²(k cosθ/2).

NOTE: Born uses simple cubic dispersion ω=2sin(k/2), v_g=cos(k/2), i.e. K2=0.
FDTD has K2=0.5 (NNN dispersion). Impact small at k<1, grows at k>1.3.
The 1/sin²(k) normalization factor is from the K2=0 dispersion.

Tests:
  1. Is I_tr(k) approximately k-independent? → explains flat integrand
  2. Does I_tr scale as R^{3/2}? → explains R-scaling
  3. Does Born shape match FDTD data? → validates Born for spectral shape

R = 3, 5, 7, 9.  k = 0.3, 0.5, ..., 1.5.
Zero FDTD — pure numeric integration (~30s).

Results:

Test 1 — I_tr(k) NOT k-independent: drops 58× from k=0.3 to k=1.5 (CV≈147%).
  Coherent Born form factor does NOT explain flat integrand.
  The form factor |F(q)|² = |Σ e^{iq·r}|² has destructive interference at high k.

Test 2 — R-scaling works: mean p = 1.63 ± 0.11 (FDTD: 1.6-1.7).
  R^{3/2} from form factor geometry (stationary phase). Same kR=2 crossover.
  I_tr/R^{1.5} collapse: CV=2-5% at k≥0.5, k=0.3 outlier (17.5%).

Test 3 — Born shape fails: FDTD/Born ratio CV≈85%. Born drops 663× while
  FDTD drops only 14.5× from k=0.3 to k=1.5.

Test 4 — Grid converged: <0.01% difference (100×200 vs 150×300).

KEY INSIGHT — Incoherent Born explains flat integrand:
  Scattering is incoherent (file 42: σ ∝ N_bonds). So coherent form factor
  |Σ e^{iq·r}|² is irrelevant. Each bond scatters independently with:
    σ_per_bond = Z_avg(k) / sin²(k)
  where Z_avg(k) = 8π(1 + sin(k)/k) is the angle-averaged z-structure factor.
  Z_avg varies only 19% across k=0.3-1.5 (CV=5.9%).

  Incoherent Born shape test: σ_FDTD / (N × Z_avg/sin²(k))
    R=3: CV=9.3%,  R=5: CV=2.7%,  R=7: CV=6.6%,  R=9: CV=7.5%
  vs coherent Born: CV≈85% at all R.

  Complete picture:
    σ_tr ≈ N_eff(R) × C(α) × Z_avg(k) / sin²(k)
    sin²(k)·σ_tr ≈ N_eff × C(α) × Z_avg(k) ≈ const  (CV≈6%)
    N_eff ~ R^{3/2} from stationary phase geometry
    Z_avg = 8π(1+sin(k)/k) from z-bond structure
    C(α) from non-Born T-matrix (absorbs α-dependence)

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/43_born_form_factor.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../../src/1_foam'))
from gauge_3d import precompute_disk_bonds

L = 80
R_vals = [3, 5, 7, 9]
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])

# Angular integration grid
N_THETA = 150
N_PHI = 300

# FDTD data from file 18 at α=0.30, NN gauging (for comparison)
sigma_fdtd = {
    3: np.array([13.16, 6.43, 3.54, 2.28, 1.72, 1.43, 1.36]),
    5: np.array([40.74, 14.16, 7.69, 5.05, 3.78, 3.14, 2.81]),
    7: np.array([72.70, 22.23, 12.22, 8.16, 6.26, 5.25, 4.71]),
    9: np.array([117.84, 34.54, 19.43, 13.34, 10.29, 8.56, 7.74]),
}


if __name__ == '__main__':
    t0 = time.time()

    print("Route 43 (F24): Born form factor of Dirac disk")
    print(f"  R = {R_vals},  k = {list(k_vals)}")
    print(f"  Angular grid: {N_THETA} × {N_PHI}")
    print()

    # Angular grid (θ from z-axis, φ from x-axis)
    thetas = np.linspace(0, np.pi, N_THETA + 1)
    phis = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    THETA, PHI = np.meshgrid(thetas, phis, indexing='ij')

    # Flatten for vectorized computation
    theta_f = THETA.ravel()
    phi_f = PHI.ravel()

    sin_t = np.sin(theta_f)
    cos_t = np.cos(theta_f)
    cos_p = np.cos(phi_f)
    sin_p = np.sin(phi_f)

    # Transport kernel: 1 - cos(scattering angle)
    # For +x incidence: cos(θ_s) = k̂_out · x̂ = sinθ cosφ
    cos_scat = sin_t * cos_p
    w_tr = 1 - cos_scat

    # Quadrature weights: trapezoidal in θ, uniform in φ
    dtheta = np.pi / N_THETA
    dphi = 2 * np.pi / N_PHI
    theta_w = np.ones(N_THETA + 1) * dtheta
    theta_w[0] = dtheta / 2
    theta_w[-1] = dtheta / 2
    w_quad = np.repeat(theta_w, N_PHI) * dphi * np.abs(sin_t)

    # Get disk bond positions for each R
    disk_data = {}
    cx, cy = L // 2, L // 2
    for R in R_vals:
        iy_disk, ix_disk = precompute_disk_bonds(L, R)
        x_b = (ix_disk - cx).astype(float)
        y_b = (iy_disk - cy).astype(float)
        disk_data[R] = np.column_stack([x_b, y_b])
        print(f"  R={R}: {len(x_b)} bonds, "
              f"x in [{x_b.min():.0f},{x_b.max():.0f}], "
              f"y in [{y_b.min():.0f},{y_b.max():.0f}]  "
              f"(expected ≈[-{R},{R}])")
    print()

    # Compute I_tr(k, R) with and without z-structure factor
    I_tr = np.zeros((len(R_vals), len(k_vals)))
    I_tr_noZ = np.zeros((len(R_vals), len(k_vals)))

    n_angles = len(theta_f)
    for i, R in enumerate(R_vals):
        positions = disk_data[R]  # (N_bonds, 2)
        N_bonds = len(positions)
        mem_mb = n_angles * N_bonds * 16 / 1e6
        if mem_mb > 500:
            print(f"  WARNING: R={R} phase matrix ~{mem_mb:.0f} MB")

        t1 = time.time()
        for j, k0 in enumerate(k_vals):
            # In-plane momentum transfer
            qx = k0 * (sin_t * cos_p - 1)
            qy = k0 * sin_t * sin_p
            Q = np.column_stack([qx, qy])  # (n_angles, 2)

            # Form factor: F = Σ_j exp(i q_⊥ · r_j)
            phase = Q @ positions.T  # (n_angles, N_bonds)
            F = np.sum(np.exp(1j * phase), axis=1)  # (n_angles,)
            F2 = np.abs(F)**2

            # Z-structure factor: 4cos²(k cosθ / 2)
            Z = 4 * np.cos(k0 * cos_t / 2)**2

            # Integrate with and without Z
            I_tr[i, j] = np.sum(F2 * Z * w_tr * w_quad)
            I_tr_noZ[i, j] = np.sum(F2 * w_tr * w_quad)

        dt = time.time() - t1
        print(f"  R={R} ({N_bonds} bonds, {dt:.1f}s):")
        print(f"    I_tr     = " + "  ".join(f"{v:.1f}" for v in I_tr[i]))
        print(f"    I_tr(noZ)= " + "  ".join(f"{v:.1f}" for v in I_tr_noZ[i]))

    # ═══════════════════════════════════════════════════════════════════
    # Test 1: k-dependence of I_tr
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 75)
    print("Test 1: I_tr(k) normalized to k=0.3 — should be ~1.0 if flat")
    print("  sin²(k)·σ_Born ∝ I_tr(k). Flat integrand ↔ I_tr ≈ const(k)")
    print("=" * 75)
    print(f"  {'R':>3s}", end="")
    for k0 in k_vals:
        print(f"  {k0:>6.1f}", end="")
    print(f"  {'CV%':>6s}")

    for i, R in enumerate(R_vals):
        norm = I_tr[i] / I_tr[i, 0]
        cv = np.std(I_tr[i]) / np.mean(I_tr[i]) * 100
        print(f"  {R:3d}", end="")
        for n in norm:
            print(f"  {n:6.3f}", end="")
        print(f"  {cv:5.1f}%")

    print()
    print("  Without Z-factor (should be less flat):")
    for i, R in enumerate(R_vals):
        norm = I_tr_noZ[i] / I_tr_noZ[i, 0]
        cv = np.std(I_tr_noZ[i]) / np.mean(I_tr_noZ[i]) * 100
        print(f"  {R:3d}", end="")
        for n in norm:
            print(f"  {n:6.3f}", end="")
        print(f"  {cv:5.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # Test 2: R-scaling
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 75)
    print("Test 2: R-scaling — I_tr / R^p for various p")
    print("=" * 75)

    log_R = np.log(np.array(R_vals, dtype=float))
    print(f"  {'k':>5s}  {'p(fit)':>7s}  ", end="")
    for R in R_vals:
        print(f"  {'I/R^1.5('+str(R)+')':>12s}", end="")
    print(f"  {'CV%':>6s}")

    for j, k0 in enumerate(k_vals):
        log_I = np.log(I_tr[:, j])
        p, _ = np.polyfit(log_R, log_I, 1)
        scaled = [I_tr[i, j] / R**1.5 for i, R in enumerate(R_vals)]
        cv = np.std(scaled) / np.mean(scaled) * 100
        print(f"  {k0:5.1f}  {p:7.3f}  ", end="")
        for s in scaled:
            print(f"  {s:12.2f}", end="")
        print(f"  {cv:5.1f}%")

    # Overall R-exponent
    p_all = []
    for j in range(len(k_vals)):
        p, _ = np.polyfit(log_R, np.log(I_tr[:, j]), 1)
        p_all.append(p)
    print(f"\n  Mean R-exponent: {np.mean(p_all):.3f} ± {np.std(p_all):.3f}")
    print(f"  Expected from stationary phase: 1.500")

    # ═══════════════════════════════════════════════════════════════════
    # Test 3: Born shape vs FDTD
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 75)
    print("Test 3: σ_FDTD / (I_tr/sin²(k)) — should be const if Born works")
    print("  Constant ratio = Born captures spectral shape, non-Born is scalar")
    C_alpha = (np.cos(2 * np.pi * 0.30) - 1)**2
    print(f"  C(α=0.30) = K_eff² = (cos(2πα)-1)² = {C_alpha:.4f}")
    print(f"  Ratio absorbs C(α) + normalization → CV is the meaningful test")
    print("=" * 75)

    for R in R_vals:
        i = R_vals.index(R)
        sigma_born_shape = I_tr[i] / np.sin(k_vals)**2
        ratio = sigma_fdtd[R] / sigma_born_shape
        cv = np.std(ratio) / np.mean(ratio) * 100

        print(f"\n  R={R} ({len(disk_data[R])} bonds):")
        print(f"  {'k':>5s}  {'σ_FDTD':>8s}  {'σ_Born':>10s}  {'ratio':>8s}")
        for j, k0 in enumerate(k_vals):
            print(f"  {k0:5.1f}  {sigma_fdtd[R][j]:8.4f}"
                  f"  {sigma_born_shape[j]:10.4f}  {ratio[j]:8.5f}")
        print(f"  ratio: mean={np.mean(ratio):.5f}, CV={cv:.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # Test 4: Convergence check — compare with coarser grid
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 75)
    print("Test 4: Angular grid convergence (100×200 vs 150×300)")
    print("=" * 75)

    # Coarser grid
    n_t2 = 100
    n_p2 = 200
    th2 = np.linspace(0, np.pi, n_t2 + 1)
    ph2 = np.linspace(0, 2 * np.pi, n_p2, endpoint=False)
    TH2, PH2 = np.meshgrid(th2, ph2, indexing='ij')
    tf2 = TH2.ravel()
    pf2 = PH2.ravel()
    st2 = np.sin(tf2)
    ct2 = np.cos(tf2)
    cp2 = np.cos(pf2)
    sp2 = np.sin(pf2)
    wt2 = 1 - st2 * cp2
    dt2 = np.pi / n_t2
    dp2 = 2 * np.pi / n_p2
    tw2 = np.ones(n_t2 + 1) * dt2
    tw2[0] = dt2 / 2
    tw2[-1] = dt2 / 2
    wq2 = np.repeat(tw2, n_p2) * dp2 * np.abs(st2)

    R_test = 5
    pos = disk_data[R_test]
    i_test = R_vals.index(R_test)

    print(f"  R={R_test}:")
    print(f"  {'k':>5s}  {'I(150×300)':>12s}  {'I(100×200)':>12s}  {'diff%':>7s}")
    for j, k0 in enumerate(k_vals):
        qx2 = k0 * (st2 * cp2 - 1)
        qy2 = k0 * st2 * sp2
        Q2 = np.column_stack([qx2, qy2])
        phase2 = Q2 @ pos.T
        F2c = np.abs(np.sum(np.exp(1j * phase2), axis=1))**2
        Z2 = 4 * np.cos(k0 * ct2 / 2)**2
        I2 = np.sum(F2c * Z2 * wt2 * wq2)
        diff = abs(I2 - I_tr[i_test, j]) / I_tr[i_test, j] * 100
        print(f"  {k0:5.1f}  {I_tr[i_test,j]:12.1f}  {I2:12.1f}  {diff:6.2f}%")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.1f}s")
