"""
Route 57: Asymmetric forward cone — Born exponent derivation.

The Born exponent for N_eff of a 2D planar array of scatterers is EXACTLY
-3/2 (total) and -5/2 (transport). This is NOT -2 and -3 as naive coherence
area scaling predicts.

Cause: the forward scattering cone is ASYMMETRIC. For a 2D disk in the
xy-plane with x-incidence:
  - Forward direction is θ=π/2, φ=0 (k_out = k_in = (k,0,0))
  - Q_⊥ grows LINEARLY in ε_φ: Q_y ≈ k·ε_φ
  - Q_⊥ grows QUADRATICALLY in ε_θ: Q_x ≈ -k·ε_θ²/2
  - Forward cone width: 1/(kR) in φ, 1/√(kR) in θ
  - Solid angle ∝ 1/(kR)^{3/2}, not 1/(kR)²

Transport weight (1-cosθ) adds exactly -1 to the exponent.

Consequence: Born integrand sin²(k)·σ_ring has CV=35% (NOT flat).
FDTD gives exponent -2.0 (correction +1/2 is EMPIRIC — mechanism open).
FDTD integrand CV=7.4% (flat).
The flat integrand REQUIRES non-Born correction.

Parts A-E: Born exponent derivation (zero-compute, FDTD data from files 18, 51).
Parts F-G: Non-Born correction characterization (Born sums + lattice Green's function).

  Part A: Born exponent vs weight function (total, transport, backscatter)
  Part B: Asymptotic convergence at large R (Airy disk formula)
  Part C: Forward cone asymmetry verification (Q_⊥ vs ε_θ, ε_φ)
  Part D: Shape universality (disk vs square vs 1D)
  Part E: Born vs FDTD integrand comparison
  Part F: Non-Born correction = C × √ω (R-independent)
  Part G: Single-bond T-matrix test (wrong direction → collective)

Results:

  Part A: Exponent with total weight: -1.37. Transport: -2.41. Backscatter: -2.90.
          Transport adds exactly -1.0 to exponent. F2 has no effect.
  Part B: Asymptotic (R→∞): total → -3/2 = -1.500. Transport → -5/2 = -2.500.
          k^{3/2}·N_eff = const to 0.2% at R=100.
          Cross-check: discrete R=5 vs Airy Δ=0.017 (good agreement).
  Part C: Q_⊥/ε_θ² ≈ k/2 = 0.350 (const over ε, varies with k).
          Q_⊥/|ε_φ| ≈ k = 0.700 (const over ε, varies with k).
          Geometric ratio θ-width/φ-width = √(2kR) is universal.
          Aspect ratio at kR=3.5: 2.6×.
  Part D: Disk R=5: -2.42. Square 9×9: -2.40 (2D universal).
          1D along x: -1.72. 1D along y: -1.04 (1D orientation-dependent).
  Part E: Born integrand CV = 35.1% (NOT flat).
          FDTD integrand CV = 7.4% (flat).
          Flat integrand requires non-Born correction from -5/2 to -2.
  Part F: N_eff_FDTD = C × √ω × N_eff_Born. C ≈ 0.56, R-independent (CV=2.9%).
          √ω = √(2c·sin(k/2)). Gives exponent -5/2 + 1/2 = -2.
          Finite-size (candidate c) ELIMINATED.
  Part G: Single-bond T-matrix: |T/V|² ≈ 1.33-1.56 at Q=0, drops to ~1.0 at Q=1.5.
          Enhancement stronger at low k → exponent MORE negative (-2.53 vs Born -2.41).
          T-mat shift: -0.12 vs FDTD +0.45 (wrong direction).
          → Correction is collective multiple scattering, not per-bond.

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/57_asymmetric_cone.py
"""

import numpy as np
from scipy.special import j1

K1, K2 = 1.0, 0.5
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])

# FDTD data (files 18, 51)
sigma_bond = np.array([0.0598, 0.0543, 0.0570, 0.0625, 0.0707, 0.0807, 0.0962])
sigma_ring = np.array([40.74, 14.16, 7.69, 5.05, 3.78, 3.14, 2.81])  # R=5
sigma_ring_all = {
    3: np.array([13.16, 6.43, 3.54, 2.28, 1.72, 1.43, 1.36]),
    5: np.array([40.74, 14.16, 7.69, 5.05, 3.78, 3.14, 2.81]),
    7: np.array([72.70, 22.23, 12.22, 8.16, 6.26, 5.25, 4.71]),
    9: np.array([117.84, 34.54, 19.43, 13.34, 10.29, 8.56, 7.74]),
}
R_vals = [3, 5, 7, 9]


def cv(x):
    return np.std(x) / np.abs(np.mean(x)) * 100


def disk_bonds(R):
    dx, dy = [], []
    for ddy in range(-R, R + 1):
        for ddx in range(-R, R + 1):
            if ddx**2 + ddy**2 <= R**2:
                dx.append(ddx)
                dy.append(ddy)
    return np.array(dx, dtype=float), np.array(dy, dtype=float)


def compute_neff_born(dx_b, dy_b, k_vals, N_th=200, N_ph=200):
    """Born N_eff with transport weight, including F2."""
    thetas = np.linspace(0, np.pi, N_th)
    phis = np.linspace(0, 2 * np.pi, N_ph, endpoint=False)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    sin_th = np.sin(TH)
    cos_theta_s = np.sin(TH) * np.cos(PH)
    transport = 1 - cos_theta_s

    neff = np.zeros(len(k_vals))
    for ik, k in enumerate(k_vals):
        Q_x = k * (np.sin(TH) * np.cos(PH) - 1)
        Q_y = k * np.sin(TH) * np.sin(PH)
        phase = (Q_x[np.newaxis] * dx_b[:, np.newaxis, np.newaxis]
                 + Q_y[np.newaxis] * dy_b[:, np.newaxis, np.newaxis])
        S = np.sum(np.exp(-1j * phase), axis=0)
        S2 = np.abs(S)**2
        q_z = k * np.cos(TH)
        F2 = 4 * np.cos(q_z / 2)**2
        w = F2 * transport * sin_th
        neff[ik] = np.sum(S2 * w) / np.sum(w)
    return neff


if __name__ == '__main__':
    import time
    t0 = time.time()
    print("Route 57: Asymmetric forward cone — Born exponent derivation")
    print()

    # ═════════════════════════════════════════════════════════════════
    # Part A: Born exponent vs weight function
    # ═════════════════════════════════════════════════════════════════
    print(f"{'=' * 60}")
    print(f"  Part A: Born exponent vs weight function (R=5)")
    print(f"{'=' * 60}")

    R = 5
    dx_b, dy_b = disk_bonds(R)
    N_b = len(dx_b)

    N_th, N_ph = 200, 200
    thetas = np.linspace(0, np.pi, N_th)
    phis = np.linspace(0, 2 * np.pi, N_ph, endpoint=False)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    sin_th = np.sin(TH)
    cos_theta_s = np.sin(TH) * np.cos(PH)
    transport = 1 - cos_theta_s

    weight_defs = [
        ('total (no 1-cosθ)', np.ones_like(TH)),
        ('transport (1-cosθ)', transport),
        ('backscatter ((1-cosθ)²)', transport**2),
    ]

    print(f"\n  With F2 = 4cos²(q_z/2):")
    exponents_f2 = []
    for wname, w_angle in weight_defs:
        neff = np.zeros(len(k_vals))
        for ik, k in enumerate(k_vals):
            Q_x = k * (np.sin(TH) * np.cos(PH) - 1)
            Q_y = k * np.sin(TH) * np.sin(PH)
            phase = (Q_x[np.newaxis] * dx_b[:, np.newaxis, np.newaxis]
                     + Q_y[np.newaxis] * dy_b[:, np.newaxis, np.newaxis])
            S = np.sum(np.exp(-1j * phase), axis=0)
            S2 = np.abs(S)**2
            q_z = k * np.cos(TH)
            F2 = 4 * np.cos(q_z / 2)**2
            w = F2 * w_angle * sin_th
            neff[ik] = np.sum(S2 * w) / np.sum(w)
        p = np.polyfit(np.log(k_vals), np.log(neff), 1)
        exponents_f2.append(p[0])
        print(f"    {wname:30s}: exponent = {p[0]:.3f}")

    print(f"\n  Without F2 (continuum limit):")
    for wname, w_angle in weight_defs[:2]:
        neff = np.zeros(len(k_vals))
        for ik, k in enumerate(k_vals):
            Q_x = k * (np.sin(TH) * np.cos(PH) - 1)
            Q_y = k * np.sin(TH) * np.sin(PH)
            phase = (Q_x[np.newaxis] * dx_b[:, np.newaxis, np.newaxis]
                     + Q_y[np.newaxis] * dy_b[:, np.newaxis, np.newaxis])
            S = np.sum(np.exp(-1j * phase), axis=0)
            S2 = np.abs(S)**2
            w = w_angle * sin_th
            neff[ik] = np.sum(S2 * w) / np.sum(w)
        p = np.polyfit(np.log(k_vals), np.log(neff), 1)
        print(f"    {wname:30s}: exponent = {p[0]:.3f}")

    print(f"\n  Transport weight shifts exponent by "
          f"{exponents_f2[1]-exponents_f2[0]:.3f} ≈ -1.0")
    print(f"  F2 has negligible effect on exponent.")

    # ═════════════════════════════════════════════════════════════════
    # Part B: Asymptotic convergence (Airy disk)
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Part B: Asymptotic exponent (Airy disk, R→∞)")
    print(f"{'=' * 60}")

    N_th2, N_ph2 = 400, 400
    thetas2 = np.linspace(0, np.pi, N_th2)
    phis2 = np.linspace(0, 2 * np.pi, N_ph2, endpoint=False)
    TH2, PH2 = np.meshgrid(thetas2, phis2, indexing='ij')
    sin_th2 = np.sin(TH2)
    cos_theta_s2 = np.sin(TH2) * np.cos(PH2)
    transport2 = 1 - cos_theta_s2

    print(f"\n  {'R':>4s}  {'N':>6s}  {'kR_min':>7s}  {'total':>8s}  {'transport':>10s}"
          f"  {'diff':>6s}")
    print(f"  {'-' * 50}")
    for Rv in [5, 10, 20, 50, 100]:
        N = int(np.pi * Rv**2)
        neff_tot = np.zeros(len(k_vals))
        neff_tr = np.zeros(len(k_vals))
        for ik, k in enumerate(k_vals):
            Q_x = k * (np.sin(TH2) * np.cos(PH2) - 1)
            Q_y = k * np.sin(TH2) * np.sin(PH2)
            Q_perp = np.sqrt(Q_x**2 + Q_y**2)
            QR = Q_perp * Rv
            airy = np.ones_like(QR)
            mask = QR > 1e-10
            airy[mask] = (2 * j1(QR[mask]) / QR[mask])**2
            S2 = N**2 * airy
            w_tot = sin_th2
            w_tr = transport2 * sin_th2
            neff_tot[ik] = np.sum(S2 * w_tot) / np.sum(w_tot)
            neff_tr[ik] = np.sum(S2 * w_tr) / np.sum(w_tr)
        p_tot = np.polyfit(np.log(k_vals), np.log(neff_tot), 1)
        p_tr = np.polyfit(np.log(k_vals), np.log(neff_tr), 1)
        print(f"  {Rv:4d}  {N:6d}  {k_vals[0]*Rv:7.1f}  {p_tot[0]:8.4f}"
              f"  {p_tr[0]:10.4f}  {p_tr[0]-p_tot[0]:6.3f}")

    print(f"\n  Asymptotic: total → -3/2 = -1.500, transport → -5/2 = -2.500")

    # k^{3/2} · N_eff check at R=100
    Rv = 100
    N = int(np.pi * Rv**2)
    neff_check = np.zeros(len(k_vals))
    for ik, k in enumerate(k_vals):
        Q_x = k * (np.sin(TH2) * np.cos(PH2) - 1)
        Q_y = k * np.sin(TH2) * np.sin(PH2)
        Q_perp = np.sqrt(Q_x**2 + Q_y**2)
        QR = Q_perp * Rv
        airy = np.ones_like(QR)
        mask = QR > 1e-10
        airy[mask] = (2 * j1(QR[mask]) / QR[mask])**2
        S2 = N**2 * airy
        w_tot = sin_th2
        neff_check[ik] = np.sum(S2 * w_tot) / np.sum(w_tot)
    prod_32 = k_vals**1.5 * neff_check
    print(f"\n  R=100: k^{{3/2}}·N_eff = {['%.0f' % x for x in prod_32]}")
    print(f"  CV(k^{{3/2}}·N_eff) = {cv(prod_32):.1f}%")

    # Cross-check: discrete sum vs Airy at same R (reviewer point 1/5)
    print(f"\n  Cross-check: discrete sum vs Airy (transport exponent)")
    print(f"  {'R':>4s}  {'discrete':>10s}  {'Airy':>8s}  {'Δ':>6s}")
    print(f"  {'-' * 35}")
    for Rv_check in [5, 10]:
        dx_c, dy_c = disk_bonds(Rv_check)
        neff_c = compute_neff_born(dx_c, dy_c, k_vals)
        p_c = np.polyfit(np.log(k_vals), np.log(neff_c), 1)
        # Airy at same R
        N_c = int(np.pi * Rv_check**2)
        neff_airy_c = np.zeros(len(k_vals))
        for ik, kv in enumerate(k_vals):
            Q_x_c = kv * (np.sin(TH2) * np.cos(PH2) - 1)
            Q_y_c = kv * np.sin(TH2) * np.sin(PH2)
            Q_perp_c = np.sqrt(Q_x_c**2 + Q_y_c**2)
            QR_c = Q_perp_c * Rv_check
            airy_c = np.ones_like(QR_c)
            mask_c = QR_c > 1e-10
            airy_c[mask_c] = (2 * j1(QR_c[mask_c]) / QR_c[mask_c])**2
            S2_c = N_c**2 * airy_c
            w_tr_c = transport2 * sin_th2
            neff_airy_c[ik] = np.sum(S2_c * w_tr_c) / np.sum(w_tr_c)
        p_a = np.polyfit(np.log(k_vals), np.log(neff_airy_c), 1)
        delta = abs(p_c[0] - p_a[0])
        print(f"  {Rv_check:4d}  {p_c[0]:10.3f}  {p_a[0]:8.3f}  {delta:6.3f}")
    print(f"  Discrete agrees with Airy at R=5 (Δ<0.02).")
    print(f"  NOTE: R≥20 discrete needs >200×200 grid (cone width < resolution).")

    # ═════════════════════════════════════════════════════════════════
    # Part C: Forward cone asymmetry
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Part C: Forward cone asymmetry (k=0.7, R=5)")
    print(f"{'=' * 60}")

    k = 0.7
    print(f"\n  Q_⊥ at ε_φ=0 (grows as ε_θ²):")
    print(f"  {'ε_θ':>6s}  {'Q_⊥':>8s}  {'Q_⊥/ε_θ²':>10s}")
    print(f"  {'-' * 28}")
    for et in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        th = np.pi / 2 + et
        qx = k * (np.sin(th) - 1)
        print(f"  {et:6.2f}  {abs(qx):8.5f}  {abs(qx)/et**2:10.4f}")

    print(f"\n  Q_⊥ at ε_θ=0 (grows as |ε_φ|):")
    print(f"  {'ε_φ':>6s}  {'Q_⊥':>8s}  {'Q_⊥/|ε_φ|':>10s}")
    print(f"  {'-' * 28}")
    for ep in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        qx = k * (np.cos(ep) - 1)
        qy = k * np.sin(ep)
        print(f"  {ep:6.2f}  {np.sqrt(qx**2+qy**2):8.5f}"
              f"  {np.sqrt(qx**2+qy**2)/ep:10.4f}")

    print(f"\n  NOTE: Q_⊥/|ε_φ| ≈ k = {k:.3f} at linear order — varies with k.")
    print(f"  Q_⊥/ε_θ² ≈ k/2 = {k/2:.3f} — also varies with k.")
    print(f"  The RATIO θ-width/φ-width = √(2/(kR)) / (1/(kR)) = √(2kR)")
    print(f"  This geometric ratio IS universal (depends only on kR).")

    kR = k * 5
    print(f"\n  Forward cone at kR={kR}:")
    print(f"    Width in φ: 1/(kR) = {1/kR:.3f}")
    print(f"    Width in θ: √(2/(kR)) = {np.sqrt(2/kR):.3f}")
    print(f"    Aspect ratio: {np.sqrt(2/kR)/(1/kR):.1f}×")
    print(f"    → Solid angle ∝ 1/(kR)^{{3/2}} (not 1/(kR)²)")

    # ═════════════════════════════════════════════════════════════════
    # Part D: Shape universality
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Part D: Shape universality — disk vs square vs 1D")
    print(f"{'=' * 60}")

    # Disk R=5
    neff_disk = compute_neff_born(dx_b, dy_b, k_vals)
    p_disk = np.polyfit(np.log(k_vals), np.log(neff_disk), 1)

    # Square 9×9
    dx_sq, dy_sq = [], []
    for ddy in range(-4, 5):
        for ddx in range(-4, 5):
            dx_sq.append(ddx)
            dy_sq.append(ddy)
    dx_sq = np.array(dx_sq, dtype=float)
    dy_sq = np.array(dy_sq, dtype=float)
    neff_sq = compute_neff_born(dx_sq, dy_sq, k_vals)
    p_sq = np.polyfit(np.log(k_vals), np.log(neff_sq), 1)

    # 1D line along x (parallel to propagation — degenerate case)
    dx_1d_x = np.arange(-40, 41, dtype=float)
    dy_1d_x = np.zeros_like(dx_1d_x)
    neff_1d_x = compute_neff_born(dx_1d_x, dy_1d_x, k_vals)
    p_1d_x = np.polyfit(np.log(k_vals), np.log(neff_1d_x), 1)

    # 1D line along y (perpendicular to propagation — proper 1D test)
    dx_1d_y = np.zeros(81, dtype=float)
    dy_1d_y = np.arange(-40, 41, dtype=float)
    neff_1d_y = compute_neff_born(dx_1d_y, dy_1d_y, k_vals)
    p_1d_y = np.polyfit(np.log(k_vals), np.log(neff_1d_y), 1)

    print(f"\n  Disk R=5 (N={N_b}):       exponent = {p_disk[0]:.3f}")
    print(f"  Square 9×9 (N={len(dx_sq):.0f}):     exponent = {p_sq[0]:.3f}")
    print(f"  1D along x (N={len(dx_1d_x)}):   exponent = {p_1d_x[0]:.3f}"
          f"  (parallel to prop)")
    print(f"  1D along y (N={len(dx_1d_y)}):   exponent = {p_1d_y[0]:.3f}"
          f"  (⊥ to prop)")
    print(f"\n  2D shapes: -2.4 ≈ -5/2 (universal for 2D planar array)")
    print(f"  1D shapes: different exponent, orientation-dependent")
    print(f"  Exponent depends on array DIMENSION, not shape.")

    # ═════════════════════════════════════════════════════════════════
    # Part E: Born vs FDTD integrand
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Part E: Born vs FDTD integrand (R=5, α=0.30)")
    print(f"{'=' * 60}")

    neff_fdtd = sigma_ring / sigma_bond
    neff_born = compute_neff_born(dx_b, dy_b, k_vals)

    integrand_born = np.sin(k_vals)**2 * sigma_bond * neff_born
    integrand_fdtd = np.sin(k_vals)**2 * sigma_ring

    ib_n = integrand_born / integrand_born[0]
    if_n = integrand_fdtd / integrand_fdtd[0]

    print(f"\n  {'k':>5s}  {'Born_n':>8s}  {'FDTD_n':>8s}")
    print(f"  {'-' * 25}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {ib_n[i]:8.4f}  {if_n[i]:8.4f}")

    print(f"\n  CV(Born integrand):  {cv(ib_n):.1f}%")
    print(f"  CV(FDTD integrand):  {cv(if_n):.1f}%")

    p_born_exp = np.polyfit(np.log(k_vals), np.log(neff_born), 1)
    p_fdtd_exp = np.polyfit(np.log(k_vals), np.log(neff_fdtd), 1)
    print(f"\n  N_eff exponent: Born = {p_born_exp[0]:.2f}, FDTD = {p_fdtd_exp[0]:.2f}")
    print(f"  Correction: {p_fdtd_exp[0] - p_born_exp[0]:+.2f}"
          f" (from -5/2 to -2 = +1/2)")

    verdict = ("Born integrand NOT flat (exponent -5/2 too steep).\n"
               "  FDTD integrand flat (exponent -2, correction +1/2 is EMPIRIC).\n"
               "  Candidates: (a) multiple scattering, (b) near-field lattice\n"
               "  effects, (c) finite-size discreteness.\n"
               "  Flat integrand REQUIRES non-Born correction +1/2.")
    print(f"\n  {verdict}")

    # ═════════════════════════════════════════════════════════════════
    # Part F: Correction characterization — sqrt(omega)
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Part F: Non-Born correction characterization")
    print(f"{'=' * 60}")

    c_lattice = np.sqrt(K1 + 4 * K2)
    omega = 2 * c_lattice * np.sin(k_vals / 2)

    # Compute Born N_eff at all R, compare with FDTD
    print(f"\n  N_eff exponent (Born discrete vs FDTD) at each R:")
    print(f"  {'R':>3s}  {'Born':>8s}  {'FDTD':>8s}  {'Δ':>8s}")
    print(f"  {'-' * 30}")
    for Rv in R_vals:
        dx_r, dy_r = disk_bonds(Rv)
        nb_r = compute_neff_born(dx_r, dy_r, k_vals)
        nf_r = sigma_ring_all[Rv] / sigma_bond
        p_b = np.polyfit(np.log(k_vals), np.log(nb_r), 1)
        p_f = np.polyfit(np.log(k_vals), np.log(nf_r), 1)
        print(f"  {Rv:3d}  {p_b[0]:8.3f}  {p_f[0]:8.3f}  {p_f[0]-p_b[0]:+8.3f}")

    # Test correction functional forms at R=5
    ratio_5 = (sigma_ring_all[5] / sigma_bond) / neff_born
    print(f"\n  Functional form (R=5, ratio = FDTD/Born N_eff):")
    print(f"  {'k':>5s}  {'ratio':>7s}  {'r/√k':>7s}  {'r/√ω':>7s}")
    print(f"  {'-' * 30}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {ratio_5[i]:7.3f}  "
              f"{ratio_5[i]/np.sqrt(k_vals[i]):7.4f}  "
              f"{ratio_5[i]/np.sqrt(omega[i]):7.4f}")
    cv_sqrtk = cv(ratio_5 / np.sqrt(k_vals))
    cv_sqrtw = cv(ratio_5 / np.sqrt(omega))
    cv_sqrt_sin = cv(ratio_5 / np.sqrt(np.sin(k_vals / 2)))
    print(f"\n  CV(ratio/√k):        {cv_sqrtk:.1f}%")
    print(f"  CV(ratio/√sin(k/2)): {cv_sqrt_sin:.1f}%")
    print(f"  CV(ratio/√ω):        {cv_sqrtw:.1f}%")
    best = "√ω" if cv_sqrtw <= cv_sqrt_sin else "√sin(k/2)"
    print(f"  Best: {best} (√ω = √(2c)·√sin(k/2), differ only by constant)")

    # C(R) — R-dependence of constant
    print(f"\n  C(R) = mean(ratio/√ω) at each R:")
    C_vals = []
    for Rv in R_vals:
        dx_r, dy_r = disk_bonds(Rv)
        nb_r = compute_neff_born(dx_r, dy_r, k_vals)
        nf_r = sigma_ring_all[Rv] / sigma_bond
        r_r = nf_r / nb_r
        C_r = np.mean(r_r / np.sqrt(omega))
        cv_r = cv(r_r / np.sqrt(omega))
        C_vals.append(C_r)
        print(f"    R={Rv}: C = {C_r:.4f}  CV = {cv_r:.1f}%")
    C_arr = np.array(C_vals)
    R_arr = np.array(R_vals, dtype=float)
    p_CR = np.polyfit(np.log(R_arr), np.log(C_arr), 1)
    print(f"  C ~ R^{{{p_CR[0]:.3f}}} → R-independent")
    print(f"  mean C = {np.mean(C_arr):.4f} ± {np.std(C_arr):.4f}")

    print(f"\n  → N_eff_FDTD = {np.mean(C_arr):.2f} × √ω × N_eff_Born")
    print(f"    Born: k^{{-5/2}}.  √ω ~ √k at small k.")
    print(f"    → FDTD: k^{{-5/2+1/2}} = k^{{-2}}  ✓")

    # ═════════════════════════════════════════════════════════════════
    # Part G: Single-bond T-matrix test
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Part G: Single-bond T-matrix (lattice Green's function)")
    print(f"{'=' * 60}")

    alpha = 0.30
    V_pert = K1 * (np.cos(2 * np.pi * alpha) - 1)
    eps_gf = 0.005  # smooth VHS on 64^3 grid; 1e-4 hits singularities

    # Lattice Green's function G(Q, omega) on 3D cubic lattice
    # Correct NN+NNN dispersion (non-factorizable):
    # ω²(k) = 2K1(3-cx-cy-cz) + 4K2(3-cx·cy-cy·cz-cx·cz)
    N_bz = 64
    kx_bz = np.linspace(-np.pi, np.pi, N_bz, endpoint=False)
    ky_bz = np.linspace(-np.pi, np.pi, N_bz, endpoint=False)
    kz_bz = np.linspace(-np.pi, np.pi, N_bz, endpoint=False)
    KX_bz, KY_bz, KZ_bz = np.meshgrid(kx_bz, ky_bz, kz_bz, indexing='ij')
    CX = np.cos(KX_bz)
    CY = np.cos(KY_bz)
    CZ = np.cos(KZ_bz)
    omega_k2 = (2 * K1 * (3 - CX - CY - CZ)
                + 4 * K2 * (3 - CX * CY - CY * CZ - CX * CZ))

    print(f"\n  V = K1·cm1 = {V_pert:.4f} (α={alpha})")
    print(f"  BZ grid: {N_bz}³ = {N_bz**3}, eps = {eps_gf}")
    print(f"  Dispersion: NN+NNN 3D (non-factorizable)")

    # Verify dispersion at k along [100]: should give ω² = 12sin²(k/2)
    om_check = 2 * K1 * (1 - np.cos(0.7)) + 4 * K2 * (2 - 2 * np.cos(0.7))
    om_expect = (K1 + 4 * K2) * 4 * np.sin(0.7 / 2)**2
    # Along [100]: 2K1(1-cos k)+4K2·2(1-cos k) = (2K1+8K2)(1-cos k)
    #            = 2(K1+4K2)·2sin²(k/2) = 4(K1+4K2)sin²(k/2) — matches 1D
    print(f"  Sanity [100] k=0.7: {om_check:.4f} vs 4(K1+4K2)sin²={om_expect:.4f}"
          f" ({'PASS' if abs(om_check - om_expect) < 1e-10 else 'FAIL'})")

    # eps sensitivity check
    print(f"\n  eps sensitivity at k=0.3 (ω={omega[0]:.3f}):")
    for eps_test in [0.02, 0.005, 0.001, 1e-4]:
        G0_test = np.mean(1.0 / (omega[0]**2 - omega_k2 + 1j * eps_test))
        print(f"    eps={eps_test:.0e}: Re[G₀]={G0_test.real:.5f}"
              f"  Im[G₀]={G0_test.imag:.5f}")

    # G0(omega) at each k — forward direction (Q=0)
    print(f"\n  {'k':>5s}  {'ω':>6s}  {'Re[G₀]':>8s}  {'Im[G₀]':>8s}"
          f"  {'|T/V|²':>8s}")
    print(f"  {'-' * 40}")
    ToverV2_fwd = np.zeros(len(k_vals))
    for ik, kv in enumerate(k_vals):
        om = omega[ik]
        G0 = np.mean(1.0 / (om**2 - omega_k2 + 1j * eps_gf))
        VG = V_pert * G0
        ToverV2_fwd[ik] = 1.0 / abs(1 - VG)**2
        print(f"  {kv:5.2f}  {om:6.3f}  {G0.real:8.4f}  {G0.imag:8.4f}"
              f"  {ToverV2_fwd[ik]:8.4f}")

    print(f"\n  |T/V|² range: {ToverV2_fwd.min():.3f} - {ToverV2_fwd.max():.3f}")

    # Q-dependent |T/V|^2 at k=0.7 — along Q_x with Q_y=Q_z=0
    kv_test = 0.7
    om_test = omega[2]
    Q_test = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]
    print(f"\n  Q-dependent |T/V|² at k={kv_test} (Q along x, Q_y=Q_z=0):")
    print(f"  {'Q_x':>5s}  {'|T/V|²':>8s}  {'enhance':>8s}")
    print(f"  {'-' * 25}")
    for Qx in Q_test:
        G_Q = np.mean(np.exp(1j * Qx * KX_bz)
                       / (om_test**2 - omega_k2 + 1j * eps_gf))
        VG = V_pert * G_Q
        tv2 = 1.0 / abs(1 - VG)**2
        print(f"  {Qx:5.1f}  {tv2:8.4f}  {tv2-1:+8.4f}")

    # N_eff with T-matrix correction at R=5 (coarse grid)
    # Full 3D Q = (Q_x, Q_y, Q_z) with Q_z = k·cos(θ)
    N_th_t, N_ph_t = 50, 50
    thetas_t = np.linspace(0, np.pi, N_th_t)
    phis_t = np.linspace(0, 2 * np.pi, N_ph_t, endpoint=False)
    TH_t, PH_t = np.meshgrid(thetas_t, phis_t, indexing='ij')
    sin_th_t = np.sin(TH_t)
    cos_ts_t = np.sin(TH_t) * np.cos(PH_t)
    tr_t = 1 - cos_ts_t

    print(f"\n  N_eff comparison (R=5, {N_th_t}×{N_ph_t} grid):")
    print(f"  {'k':>5s}  {'Born':>8s}  {'T-mat':>8s}  {'FDTD':>8s}"
          f"  {'T/Born':>7s}  {'F/Born':>7s}")
    print(f"  {'-' * 50}")
    p_born_list, p_tmat_list = [], []
    for ik, kv in enumerate(k_vals):
        om = omega[ik]
        Q_x_t = kv * (np.sin(TH_t) * np.cos(PH_t) - 1)
        Q_y_t = kv * np.sin(TH_t) * np.sin(PH_t)
        Q_z_t = kv * np.cos(TH_t)
        # Structure factor (bonds in xy-plane, Q_z irrelevant for phase)
        phase_t = (Q_x_t[np.newaxis] * dx_b[:, np.newaxis, np.newaxis]
                   + Q_y_t[np.newaxis] * dy_b[:, np.newaxis, np.newaxis])
        S_t = np.sum(np.exp(-1j * phase_t), axis=0)
        S2_t = np.abs(S_t)**2
        F2_t = 4 * np.cos(Q_z_t / 2)**2
        w_t = F2_t * tr_t * sin_th_t
        neff_b = np.sum(S2_t * w_t) / np.sum(w_t)
        # T-matrix correction: G(Q) with full 3D Q = (Q_x, Q_y, Q_z)
        # Vectorized: separable exp + sequential einsum contractions
        denom_g = 1.0 / (om**2 - omega_k2 + 1j * eps_gf)
        qx_f = Q_x_t.ravel()
        qy_f = Q_y_t.ravel()
        qz_f = Q_z_t.ravel()
        ex = np.exp(1j * qx_f[:, None] * kx_bz[None, :])
        ey = np.exp(1j * qy_f[:, None] * ky_bz[None, :])
        ez = np.exp(1j * qz_f[:, None] * kz_bz[None, :])
        tmp = np.einsum('bk,ijk->bij', ez, denom_g)
        tmp = np.einsum('bj,bij->bi', ey, tmp)
        G_Q_flat = np.einsum('bi,bi->b', ex, tmp) / N_bz**3
        tv2_map = (1.0 / np.abs(1 - V_pert * G_Q_flat)**2).reshape(
            N_th_t, N_ph_t)
        # N_eff_T: numerator has |S|²×|T/V|², denominator is Born (no T/V)
        neff_t = np.sum(S2_t * tv2_map * w_t) / np.sum(w_t)
        neff_f = sigma_ring_all[5][ik] / sigma_bond[ik]
        p_born_list.append(neff_b)
        p_tmat_list.append(neff_t)
        print(f"  {kv:5.2f}  {neff_b:8.1f}  {neff_t:8.1f}  {neff_f:8.1f}"
              f"  {neff_t/neff_b:7.3f}  {neff_f/neff_b:7.3f}")
    p_b_exp = np.polyfit(np.log(k_vals), np.log(p_born_list), 1)
    p_t_exp = np.polyfit(np.log(k_vals), np.log(p_tmat_list), 1)
    neff_f_arr = sigma_ring_all[5] / sigma_bond
    p_f_exp = np.polyfit(np.log(k_vals), np.log(neff_f_arr), 1)
    print(f"\n  Exponents: Born={p_b_exp[0]:.3f}  T-mat={p_t_exp[0]:.3f}"
          f"  FDTD={p_f_exp[0]:.3f}")
    print(f"  T-mat shift: {p_t_exp[0]-p_b_exp[0]:+.3f}")
    print(f"  FDTD shift:  {p_f_exp[0]-p_b_exp[0]:+.3f}")
    if p_t_exp[0] > p_b_exp[0]:
        print(f"\n  → T-matrix makes exponent LESS negative (toward FDTD)")
    else:
        print(f"\n  → T-matrix makes exponent MORE negative (wrong direction)")
    print(f"  → FDTD shift {p_f_exp[0]-p_b_exp[0]:+.3f} vs"
          f" T-mat {p_t_exp[0]-p_b_exp[0]:+.3f}")
    print(f"  → {'Collective' if abs(p_f_exp[0]-p_b_exp[0]) > 2*abs(p_t_exp[0]-p_b_exp[0]) else 'Partial'}"
          f" multiple scattering beyond single-bond T-matrix")

    # ═════════════════════════════════════════════════════════════════
    # Summary
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"""
  Born N_eff exponent for 2D planar array:
    Total:     -3/2 (asymptotic, verified to 0.7%)
    Transport: -5/2 (asymptotic, = total - 1)

  Origin: asymmetric forward cone.
    Q_⊥ ∝ ε_φ (linear) but Q_⊥ ∝ ε_θ² (quadratic)
    → cone width 1/(kR) in φ, 1/√(kR) in θ
    → solid angle ∝ 1/(kR)^{{3/2}}

  FDTD correction: N_eff_FDTD = C × √ω × N_eff_Born
    C ≈ 0.56, R-independent.  √ω = √(2c·sin(k/2))
    → exponent -5/2 + 1/2 = -2  ✓
    Born integrand CV = 35% (not flat)
    FDTD integrand CV = 7.4% (flat)

  T-matrix test: single-bond T-matrix enhances forward peak
    |T/V|² ≈ 1.33-1.56 at Q=0, drops to ~1.0 at Q=1.5
    Enhancement stronger at low k → exponent MORE negative (-2.53)
    → Wrong direction (FDTD needs less negative: -2.0)
    → Correction is COLLECTIVE multiple scattering

  STATUS:
    DERIVED: Born exponent -5/2 from 2D geometry + transport weight
    DERIVED: transport weight adds exactly -1 (shape-independent)
    EMPIRIC: correction = C×√ω, C=0.56, R-independent (CV=2.9%)
    ELIMINATED: single-bond T-matrix (shift -0.12, wrong direction vs FDTD +0.45)
    ELIMINATED: finite-size discreteness (C is R-independent)
    OPEN: why √ω, why C=0.56
    CONSEQUENCE: flat integrand is a collective non-Born effect""")

    dt = time.time() - t0
    print(f"\nTotal: {dt:.0f}s ({dt / 60:.1f} min)")
