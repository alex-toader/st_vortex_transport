"""
Route 54 (I17): Vectorial T-matrix for single z-bond.

Question: WHY is σ_xx (same-pol) ≈ const(k)?
File 52: scalar T-matrix FAILS (|DK·G|≪1, Born).
File 53: σ_xx≈const (CV≈62-66%), compensation hypothesis FAILS.

The Peierls coupling R(2πα)-I is a 2×2 matrix on (ux, uy):
  ΔK = K1 × [[cm1, -s_phi], [s_phi, cm1]]
where cm1 = cos(2πα)-1, s_phi = sin(2πα).

Scalar T-matrix used only the diagonal (cm1) → Born.
But the full 2×2 problem couples x↔y channels.

Method: 2×2 T-matrix on lattice Green's function.
  G_anti(ω) = ∫ dk³/(2π)³ × (1-cos k_z) / (ω² - ω²(k) + iε)
  G_anti is scalar (same for both polarizations on isotropic lattice).
  T = ΔK · (I - ΔK · G_anti)^{-1}   [2×2 matrix]
  σ_xx ∝ |T_xx|² × Z_avg/sin²(k)
  σ_xy ∝ |T_xy|² × Z_avg/sin²(k)
  σ_tot ∝ (|T_xx|² + |T_xy|²) × Z_avg/sin²(k)

Key test: does |T_xx|² ∝ sin²(k) → σ_xx ≈ const?

NOTE: ΔK · G_anti is a 2×2 matrix with structure [[cm1·g, -s·g], [s·g, cm1·g]]
where g = K1·G_anti. The eigenvalues are g·(cm1 ± i·s) = g·e^{±i·2πα}.
So |eigenvalue| = |g|·|e^{i·2πα} - 1| = |g|·√(cm1² + s²) = |g|·√(2(1-cos2πα)).
At α=0.30: √(2·2.309) = 2.148, so |eigenvalue| = 2.148×|g| vs scalar |cm1·g|=1.309×|g|.
The vectorial coupling is stronger by factor 2.148/1.309 = 1.64.
Still |eigenvalue| < 0.42 at max (vs scalar 0.21) — still Born?

Results (2s):

  VECTORIAL T-MATRIX ALSO FAILS. CV(σ_xx_T)=109% vs FDTD 62%.
  |T_xx|² is nearly constant (~1.02 variation). Does NOT grow as sin²(k).
  |eigenvalue| ≤ 0.24 → still Born regime. T-matrix correction is 10-20%.

  α=0.30: CV(σ_xx_T)=109.2%, CV(σ_xy_T)=101.8%. Both ≈ Born shape.
  α=0.10: CV(σ_xx_T)=115.6%, CV(σ_xy_T)=109.0%. Same failure.

  σ_xx ≈ const(k) is NOT a T-matrix effect (scalar or vectorial).
  Single-site scattering theory cannot explain it.
  Must be: (a) near-field / wave interference, (b) lattice discreteness,
  or (c) the σ ∝ |T|²×Z_avg/sin² formula is wrong for this geometry.

  KEY INSIGHT from file 53 + 54: σ_xx≈const is α-INDEPENDENT (CV≈62-66%).
  The α-threshold (α≥0.25) for flat integrand on ring comes from σ_xy
  suppression: at α≥0.25, |cm1|>|s_phi| → σ_xy is small → σ_tot ≈ σ_xx ≈ const.
  At α<0.25, |s_phi|>|cm1| → σ_xy dominates at high k → σ_tot grows.
  This is derivable analytically from Peierls geometry: |cm1|=|s_phi| at α=0.25.

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/54_vectorial_tmatrix.py
"""

import numpy as np
import time

K1, K2 = 1.0, 0.5
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])

# FDTD data from file 53 (α=0.30)
sigma_xx_fdtd = np.array([0.0595, 0.0535, 0.0553, 0.0593, 0.0651, 0.0713, 0.0804])
sigma_xy_fdtd = np.array([0.0003, 0.0008, 0.0017, 0.0032, 0.0056, 0.0094, 0.0158])
sigma_tot_fdtd = sigma_xx_fdtd + sigma_xy_fdtd
# Sanity: sum matches file 51
sigma_51 = np.array([0.0598, 0.0543, 0.0571, 0.0624, 0.0707, 0.0807, 0.0962])
_max_diff = np.max(np.abs(sigma_tot_fdtd - sigma_51) / sigma_51) * 100
assert _max_diff < 2.0, f"FDTD data inconsistent with file 51: {_max_diff:.1f}%"


def omega_sq(kx, ky, kz):
    """Dispersion ω²(k) for NN (K1) + NNN (K2) simple cubic."""
    nn = 2 * K1 * ((1 - np.cos(kx)) + (1 - np.cos(ky)) + (1 - np.cos(kz)))
    nnn = 4 * K2 * ((1 - np.cos(kx) * np.cos(ky))
                   + (1 - np.cos(ky) * np.cos(kz))
                   + (1 - np.cos(kx) * np.cos(kz)))
    return nn + nnn


def compute_G_anti(omega2_target, N_grid=200, eps=1e-4):
    """G_anti(ω²) = ∫ dk³/(2π)³ × (1-cos k_z) / (ω² - ω²(k) + iε)"""
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
    print("Route 54: Vectorial T-matrix single bond (2×2)")

    # Dispersion
    omega2_vals = omega_sq(k_vals, 0, 0)

    # G_anti is scalar: G_xy = 0 by cubic symmetry (integrand odd in kx↔ky).
    # K1=2K2 ensures isotropy. Both polarizations see the same Green's function.
    print(f"\n  Computing G_anti (N=200, eps=1e-4)...")
    G_anti = np.zeros(len(k_vals), dtype=complex)
    for i, w2 in enumerate(omega2_vals):
        G_anti[i] = compute_G_anti(w2, N_grid=200, eps=1e-4)
        print(f"    k={k_vals[i]:.1f}: G_anti = {G_anti[i].real:+.6f}"
              f" {G_anti[i].imag:+.6f}j")

    Z_avg = 8 * np.pi * (1 + np.sin(k_vals) / k_vals)

    # ═══════════════════════════════════════════════════════════════
    # Vectorial T-matrix at α=0.30
    # ═══════════════════════════════════════════════════════════════
    for alpha in [0.30, 0.10]:
        cm1 = np.cos(2 * np.pi * alpha) - 1
        s_phi = np.sin(2 * np.pi * alpha)
        print(f"\n{'='*60}")
        print(f"  α = {alpha}")
        print(f"  cm1 = {cm1:.4f}, s_phi = {s_phi:.4f}")
        print(f"  |R-I| = sqrt(cm1²+s²) = {np.sqrt(cm1**2 + s_phi**2):.4f}")
        print(f"{'='*60}")

        # ΔK matrix (2×2)
        DK = K1 * np.array([[cm1, -s_phi],
                            [s_phi, cm1]])
        print(f"\n  ΔK = K1 × (R-I) =")
        print(f"    [[{DK[0,0]:+.4f}, {DK[0,1]:+.4f}],")
        print(f"     [{DK[1,0]:+.4f}, {DK[1,1]:+.4f}]]")

        # Eigenvalues of ΔK
        eig_plus = K1 * (cm1 + 1j * s_phi)   # = K1 × (e^{i2πα} - 1)
        eig_minus = K1 * (cm1 - 1j * s_phi)  # = K1 × (e^{-i2πα} - 1)
        print(f"\n  ΔK eigenvalues: {abs(eig_plus):.4f} × e^{{±iφ}}")
        print(f"  |eig| = K1·√(2(1-cos2πα)) = {abs(eig_plus):.4f}")
        print(f"  Scalar |cm1| = {abs(cm1):.4f} (ratio {abs(eig_plus)/abs(cm1):.2f}×)")

        T_xx = np.zeros(len(k_vals), dtype=complex)
        T_xy = np.zeros(len(k_vals), dtype=complex)
        T2_xx = np.zeros(len(k_vals))
        T2_xy = np.zeros(len(k_vals))
        T2_tot = np.zeros(len(k_vals))

        print(f"\n  {'k':>5s}  {'|DK·G|_eig':>10s}  {'|T_xx|²':>10s}"
              f"  {'|T_xy|²':>10s}  {'|T_tot|²':>10s}")
        for i in range(len(k_vals)):
            g = G_anti[i]
            # M = I - ΔK·G_anti (2×2)
            M = np.eye(2) - DK * g
            M_inv = np.linalg.inv(M)
            T = DK @ M_inv

            T_xx[i] = T[0, 0]
            T_xy[i] = T[0, 1]
            T2_xx[i] = abs(T[0, 0])**2
            T2_xy[i] = abs(T[0, 1])**2
            T2_tot[i] = T2_xx[i] + T2_xy[i]

            eig_g = abs(eig_plus * g)
            print(f"  {k_vals[i]:5.2f}  {eig_g:10.4f}  {T2_xx[i]:10.6f}"
                  f"  {T2_xy[i]:10.6f}  {T2_tot[i]:10.6f}")

        # σ ∝ |T|² × Z_avg / sin²(k)
        sigma_xx_T = T2_xx * Z_avg / np.sin(k_vals)**2
        sigma_xy_T = T2_xy * Z_avg / np.sin(k_vals)**2
        sigma_tot_T = T2_tot * Z_avg / np.sin(k_vals)**2

        # Normalize xx and tot to first k; xy to k=0.9 (index 3) since xy≈0 at k=0.3
        sigma_xx_T_n = sigma_xx_T / sigma_xx_T[0]
        sigma_xy_T_n = sigma_xy_T / sigma_xy_T[3]  # k=0.9
        sigma_tot_T_n = sigma_tot_T / sigma_tot_T[0]

        fdtd_xx_n = sigma_xx_fdtd / sigma_xx_fdtd[0] if alpha == 0.30 else None
        fdtd_xy_n = sigma_xy_fdtd / sigma_xy_fdtd[3] if alpha == 0.30 else None  # k=0.9

        cv_xx_T = cv(sigma_xx_T_n)
        cv_xy_T = cv(sigma_xy_T_n)
        cv_tot_T = cv(sigma_tot_T_n)

        print(f"\n  Normalized σ ∝ |T|²×Z_avg/sin²(k):")
        print(f"  {'k':>5s}  {'σ_xx_T_n':>10s}  {'σ_xy_T_n':>10s}"
              f"  {'σ_tot_T_n':>10s}", end="")
        if fdtd_xx_n is not None:
            print(f"  {'FDTD_xx_n':>10s}  {'FDTD_xy_n':>10s}", end="")
        print()

        for i in range(len(k_vals)):
            line = (f"  {k_vals[i]:5.2f}  {sigma_xx_T_n[i]:10.4f}"
                    f"  {sigma_xy_T_n[i]:10.4f}  {sigma_tot_T_n[i]:10.4f}")
            if fdtd_xx_n is not None:
                line += f"  {fdtd_xx_n[i]:10.4f}  {fdtd_xy_n[i]:10.4f}"
            print(line)

        print(f"\n  CV(σ_xx_T) = {cv_xx_T:.1f}%"
              + (f"  (FDTD: {cv(fdtd_xx_n):.1f}%)" if fdtd_xx_n is not None else ""))
        print(f"  CV(σ_xy_T) = {cv_xy_T:.1f}%"
              + (f"  (FDTD: {cv(fdtd_xy_n):.1f}%)" if fdtd_xy_n is not None else ""))
        print(f"  CV(σ_tot_T) = {cv_tot_T:.1f}%")

        # Check if vectorial T-matrix captures σ_xx ≈ const
        if cv_xx_T < 30:
            print(f"\n  ==> VECTORIAL T-MATRIX EXPLAINS σ_xx ≈ const!")
        elif cv_xx_T < cv_tot_T * 0.5:
            print(f"\n  ==> Partial: vectorial T reduces CV by"
                  f" {cv_tot_T/cv_xx_T:.1f}×")
        else:
            print(f"\n  ==> Vectorial T-matrix still fails for σ_xx"
                  f" (CV={cv_xx_T:.0f}%)")

        # |T_xx|² shape: does it grow as sin²(k)?
        T2_xx_n = T2_xx / T2_xx[0]
        sin2_n = np.sin(k_vals)**2 / np.sin(k_vals[0])**2
        print(f"\n  |T_xx|² shape vs sin²(k):")
        print(f"  {'k':>5s}  {'|T_xx|²_n':>10s}  {'sin²_n':>8s}  {'ratio':>8s}")
        for i in range(len(k_vals)):
            r = T2_xx_n[i] / sin2_n[i]
            print(f"  {k_vals[i]:5.2f}  {T2_xx_n[i]:10.4f}"
                  f"  {sin2_n[i]:8.4f}  {r:8.4f}")
        cv_ratio = cv(T2_xx_n / sin2_n)
        print(f"  CV(|T_xx|²/sin²) = {cv_ratio:.1f}%")
        if cv_ratio < 10:
            print(f"  ==> |T_xx|² ∝ sin²(k)! σ_xx = |T_xx|²×Z_avg/sin² ≈ Z_avg ≈ const")

        # More precise: σ_xx = const requires |T_xx|² ∝ sin²(k)/Z_avg
        target = np.sin(k_vals)**2 / Z_avg
        target_n = target / target[0]
        cv_precise = cv(T2_xx_n / target_n)
        print(f"  CV(|T_xx|²/(sin²/Z_avg)) = {cv_precise:.1f}%"
              f"  (exact const requires ~0%)")

    # ═══════════════════════════════════════════════════════════════
    # Key insight from files 53+54
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  KEY INSIGHT")
    print(f"{'='*60}")
    print(f"  σ_xx ≈ const is α-INDEPENDENT (CV≈62-66% at both α).")
    print(f"  The α-threshold (α≥0.25) for flat integrand on ring")
    print(f"  comes from σ_xy suppression at |cm1|>|s_phi|:")
    print(f"    α=0.30: |cm1|=1.309 > |s_phi|=0.951 → σ_xy small → σ_tot ≈ σ_xx")
    print(f"    α=0.10: |cm1|=0.191 < |s_phi|=0.588 → σ_xy dominates at high k")
    print(f"    Threshold: |cm1|=|s_phi| ⟺ α=0.25 exactly.")

    t_total = time.time() - t0
    print(f"\nTotal: {t_total:.0f}s")
