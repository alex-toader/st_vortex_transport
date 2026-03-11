"""
Route 52 (I14): T-matrix single bond — Part A only (analytic).

Question: WHY is σ_bond ≈ const(k)? (file 51: CV=20%, grows only 1.6×)

Method: scalar T-matrix for one modified z-bond (ΔK = K1·cm1) on 3D cubic lattice.
  T(ω) = ΔK / (1 - ΔK · G_anti(ω))
  G_anti(ω) = ∫ dk³/(2π)³ × (1-cos k_z) / (ω² - ω²(k) + iε)
  σ_T(k) ∝ |T(ω(k))|² × Z_avg(k) / sin²(k)  [shape comparison, normalized]

NOTE: scalar T-matrix assumes single polarization channel.
File 49 shows polarization conversion grows with k (1%→63% cross-pol).
If scalar T-matrix fails to reproduce σ_bond≈const → polarization compensation
is the mechanism (σ_xx decreases + σ_xy increases ≈ const).

Results (2s):

  |DK·G_anti| ≈ 0.11-0.19 ≪ 1 → Born regime for scalar channel.
  |T|² varies only 2.15→2.63 (1.2×). T-matrix correction is 10-20%.
  σ_T ∝ |T|²×Z_avg/sin²(k) ≈ Born shape: CV=105.5%.
  FDTD: CV=20.3%.

  SCALAR T-MATRIX FAILS. Does NOT explain σ_bond ≈ const.
  |DK·G| ≪ 1 means single-channel scattering is perturbative.
  The non-Born σ_bond ≈ const must come from polarization mixing
  (cross-pol σ_xy compensating same-pol σ_xx decrease).

  α scan: |DK·G_anti| at k=0.9:
    α=0.05: 0.006, α=0.10: 0.020, α=0.20: 0.070, α=0.30: 0.135,
    α=0.40: 0.185, α=0.50: 0.207
  Never reaches |DK·G|~1. Scalar T-matrix is Born at ALL α.
  The α-threshold (α≈0.25) is NOT a T-matrix crossover.

  Convergence: N=100 vs N=200: 1.2%. ε=1e-3 vs 1e-5: Re ~3%, Im variable.
  Im(G_anti) is small (|Im/Re| < 16%) → dominated by real part.

  Next: polarization decomposition of σ_bond (file 53).

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/52_tmatrix_single_bond.py
"""

import numpy as np
import time

K1, K2 = 1.0, 0.5
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])

# FDTD σ_bond from file 51 (α=0.30, for comparison)
sigma_fdtd_030 = np.array([0.0598, 0.0543, 0.0571, 0.0624, 0.0707, 0.0807, 0.0962])


def omega_sq(kx, ky, kz):
    """Dispersion ω²(k) for NN (K1) + NNN (K2) simple cubic."""
    nn = 2 * K1 * ((1 - np.cos(kx)) + (1 - np.cos(ky)) + (1 - np.cos(kz)))
    nnn = 4 * K2 * ((1 - np.cos(kx) * np.cos(ky))
                   + (1 - np.cos(ky) * np.cos(kz))
                   + (1 - np.cos(kx) * np.cos(kz)))
    return nn + nnn


def compute_G_anti(omega2_target, N_grid=200, eps=1e-4):
    """G_anti(ω²) = ∫ dk³/(2π)³ × (1-cos k_z) / (ω² - ω²(k) + iε)
    Numeric integration on uniform grid. Octant [0,π]³ × 8 (midpoint rule)."""
    dk = np.pi / N_grid
    kx = np.linspace(dk / 2, np.pi - dk / 2, N_grid)
    ky = np.linspace(dk / 2, np.pi - dk / 2, N_grid)
    kz = np.linspace(dk / 2, np.pi - dk / 2, N_grid)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    w2 = omega_sq(KX, KY, KZ)
    numerator = 1 - np.cos(KZ)
    denom = omega2_target - w2 + 1j * eps

    vol = (dk ** 3) * 8 / (2 * np.pi) ** 3
    G = vol * np.sum(numerator / denom)
    return G


def cv(x):
    return np.std(x) / np.abs(np.mean(x)) * 100


if __name__ == '__main__':
    t0 = time.time()
    print("Route 52: T-matrix single bond (scalar)")
    print(f"  K1={K1}, K2={K2}")

    # Dispersion: ω(k) for +x propagation (ky=kz=0)
    omega2_vals = omega_sq(k_vals, 0, 0)
    omega_vals = np.sqrt(omega2_vals)
    print(f"\n  Dispersion (ky=kz=0):")
    print(f"  {'k':>5s}  {'omega2':>8s}  {'omega':>8s}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {omega2_vals[i]:8.4f}  {omega_vals[i]:8.4f}")

    # ═════════════════════════════════════════════════════════════
    # Convergence and ε tests
    # ═════════════════════════════════════════════════════════════
    print(f"\n  Convergence test (k=0.9, ω²={omega2_vals[3]:.4f}):")
    G_100 = compute_G_anti(omega2_vals[3], N_grid=100, eps=1e-4)
    G_200 = compute_G_anti(omega2_vals[3], N_grid=200, eps=1e-4)
    rel_diff = abs(G_100 - G_200) / abs(G_200) * 100
    print(f"    N=100: {G_100.real:+.6f} {G_100.imag:+.6f}j")
    print(f"    N=200: {G_200.real:+.6f} {G_200.imag:+.6f}j")
    print(f"    rel diff: {rel_diff:.2f}%")

    print(f"\n  ε sensitivity (k=0.9, N=200):")
    for eps_test in [1e-3, 1e-4, 1e-5]:
        G_test = compute_G_anti(omega2_vals[3], N_grid=200, eps=eps_test)
        print(f"    eps={eps_test:.0e}: G = {G_test.real:+.6f} {G_test.imag:+.6f}j")

    # ═════════════════════════════════════════════════════════════
    # G_anti at all k
    # ═════════════════════════════════════════════════════════════
    print(f"\n  Computing G_anti (N=200, eps=1e-4)...")
    G_anti = np.zeros(len(k_vals), dtype=complex)
    for i, w2 in enumerate(omega2_vals):
        G_anti[i] = compute_G_anti(w2, N_grid=200, eps=1e-4)
        print(f"    k={k_vals[i]:.1f}: G_anti = {G_anti[i].real:+.6f} {G_anti[i].imag:+.6f}j")

    # ═════════════════════════════════════════════════════════════
    # T-matrix at α=0.30
    # ═════════════════════════════════════════════════════════════
    alpha = 0.30
    cm1 = np.cos(2 * np.pi * alpha) - 1
    DK = K1 * cm1
    print(f"\n  alpha={alpha}, cm1={cm1:.4f}, DK={DK:.4f}")

    T = DK / (1 - DK * G_anti)
    T2 = np.abs(T) ** 2

    # σ_T ∝ |T|² × Z_avg / sin²(k)  — shape comparison only (normalized)
    Z_avg = 8 * np.pi * (1 + np.sin(k_vals) / k_vals)
    sigma_T = T2 * Z_avg / np.sin(k_vals) ** 2
    sigma_T_n = sigma_T / sigma_T[0]

    print(f"\n  {'k':>5s}  {'|T|2':>10s}  {'Z_avg':>8s}  {'sigma_T_n':>10s}  {'FDTD_n':>8s}")
    fdtd_n = sigma_fdtd_030 / sigma_fdtd_030[0]
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {T2[i]:10.6f}  {Z_avg[i]:8.3f}"
              f"  {sigma_T_n[i]:10.4f}  {fdtd_n[i]:8.4f}")

    cv_T = cv(sigma_T_n)
    cv_fdtd = cv(fdtd_n)
    print(f"\n  CV(sigma_T) = {cv_T:.1f}% (FDTD: {cv_fdtd:.1f}%)")
    print(f"  sigma_T ratio first/last = {sigma_T_n[0]/sigma_T_n[-1]:.3f}"
          f" (FDTD: {fdtd_n[0]/fdtd_n[-1]:.3f})")

    if cv_T > 50:
        print(f"\n  ==> SCALAR T-MATRIX FAILS (CV={cv_T:.0f}% vs FDTD {cv_fdtd:.0f}%)")
        print(f"      σ_bond ≈ const is NOT from scalar non-Born.")
        print(f"      Likely mechanism: polarization compensation (σ_xx + σ_xy ≈ const).")

    # ═════════════════════════════════════════════════════════════
    # Coupling strength |DK × G_anti|
    # ═════════════════════════════════════════════════════════════
    print(f"\n  Coupling strength |DK * G_anti|:")
    for i in range(len(k_vals)):
        dkg = DK * G_anti[i]
        print(f"    k={k_vals[i]:.1f}: |DK*G| = {abs(dkg):.3f}"
              f"  (Re={dkg.real:.4f}, Im={dkg.imag:.4f})")

    max_dkg = max(abs(DK * G_anti[i]) for i in range(len(k_vals)))
    print(f"  max |DK*G| = {max_dkg:.3f}")
    if max_dkg < 0.5:
        print(f"  ==> Born regime (|DK*G| ≪ 1). Scalar T-matrix ≈ Born at all k.")

    # ═════════════════════════════════════════════════════════════
    # α scan: |DK × G_anti| at k=0.9 (mid-BZ)
    # ═════════════════════════════════════════════════════════════
    print(f"\n  α scan: |DK * G_anti| at k=0.9:")
    G_mid = G_anti[3]  # k=0.9
    alpha_scan = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    print(f"  {'alpha':>6s}  {'DK':>8s}  {'|DK*G|':>8s}")
    for a in alpha_scan:
        dk_a = K1 * (np.cos(2 * np.pi * a) - 1)
        strength = abs(dk_a * G_mid)
        label = " <-- max" if a == 0.50 else ""
        print(f"  {a:6.2f}  {dk_a:8.4f}  {strength:8.3f}{label}")

    print(f"\n  |DK*G| never reaches 1. Scalar T-matrix is Born at ALL α.")
    print(f"  The α-threshold (α≈0.25 for flat integrand) is NOT")
    print(f"  a scalar T-matrix crossover.")

    # ═════════════════════════════════════════════════════════════
    # Also try with v_g² correction
    # ═════════════════════════════════════════════════════════════
    # v_g for +x propagation: ω² = 12sin²(k/2) → v_g = √3 cos(k/2)
    v_g = np.sqrt(3.0) * np.cos(k_vals / 2)
    sigma_T_vg = T2 * Z_avg / (np.sin(k_vals) ** 2 * v_g ** 2)
    sigma_T_vg_n = sigma_T_vg / sigma_T_vg[0]

    print(f"\n  With v_g² correction (σ ∝ |T|²×Z_avg/(sin²×v_g²)):")
    print(f"  {'k':>5s}  {'v_g':>6s}  {'σ_T_vg_n':>10s}  {'FDTD_n':>8s}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {v_g[i]:6.4f}  {sigma_T_vg_n[i]:10.4f}  {fdtd_n[i]:8.4f}")
    print(f"  CV(sigma_T_vg) = {cv(sigma_T_vg_n):.1f}% (worse — v_g correction increases variation)")

    t_total = time.time() - t0
    print(f"\nTotal: {t_total:.0f}s")
