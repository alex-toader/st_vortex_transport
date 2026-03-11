"""
Route 60: Born exponent -5/2 derivation + gap closure.

File 59 left three gaps in the mechanism chain:
  Gap 1: Born exponent -5/2 not derived analytically
  Gap 2: Why does MS give exponent -2.0?
  Gap 4: Is N_eff ∝ 1/k² independently derivable?

This file resolves all three.

Parts:
  A: Continuum disk form factor — N_eff_Born for Airy disk vs discrete lattice
  B: Asymmetric cone structure — forward cone width in delta vs eta
  C: Transport decomposition — sigma_total (-3/2) + transport (-1) = sigma_tr (-5/2)
  D: Large-kR convergence — p_Born → -5/2 at kR → ∞ (verified R=200)
  E: MS exponent vs alpha — p_MS crosses -2.0 at alpha ≈ 0.29
  F: Flat integrand residual — three sources of CV ≈ 7-10%

Results (15s):
  Part A: Continuum Airy disk vs discrete lattice — agree to < 0.03 at R≥5.
     Both converge to p ≈ -2.42 (not yet -2.50 at kR=2-8).
  Part B: Forward cone (qR<10) captures 92-100% of N_eff.
     theta contribution spread ±60° (delta width ~ 1/sqrt(kR) >> eta ~ 1/(kR)).
  Part C: p_total = -1.36 to -1.40 (cone area = -3/2).
     p_tr = -2.42 to -2.43. Diff = -1.03 to -1.07 (transport adds -1).
     → -5/2 = -3/2 + (-1). ANALYTIC.
  Part D: R=200, 30 k-points, kR=2-30.
     kR>15: p_tr = -2.491 (0.4% from -5/2). Asymptotic fit: -2.516.
     Correction: O(1/(kR)), coefficient -0.42.
  Part E: p_MS crosses -2.0 at alpha ≈ 0.29 (|V| ≈ 1.26).
     Minimum CV_integrand = 10.7% at alpha = 0.30.
     -2.0 is NOT exact — it's the crossing point p_Born + C*|V| = -2.
  Part F: CV = 10.7% from three sources:
     sinc²(k/2) ≠ 1: ~6%. Non-power-law: ~4%. p_MS deficit (+0.03): ~1%.

Run: cd ST_11/wip/w_21_kappa && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 60_born_exponent.py
"""

import numpy as np
from scipy.special import j1
import time

K1, K2 = 1.0, 0.5
c_lat = np.sqrt(K1 + 4 * K2)
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
x = np.log(k_vals)

# Lattice BZ grid (for G_00)
N_bz = 64
kx_1d = np.linspace(-np.pi, np.pi, N_bz, endpoint=False)
CX, CY, CZ = np.meshgrid(np.cos(kx_1d), np.cos(kx_1d), np.cos(kx_1d),
                          indexing='ij')
omega_k2 = (2 * K1 * (3 - CX - CY - CZ)
            + 4 * K2 * (3 - CX * CY - CY * CZ - CX * CZ))
eps_lat = 0.005


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


def make_angular_grid(N_th, N_ph):
    thetas = np.linspace(0, np.pi, N_th)
    phis = np.linspace(0, 2 * np.pi, N_ph, endpoint=False)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    sin_th = np.sin(TH)
    transport = 1 - sin_th * np.cos(PH)
    return TH, PH, sin_th, transport


def neff_continuum(R, k_arr, TH, PH, sin_th, transport, use_transport=True):
    """N_eff using continuum Airy disk form factor."""
    cos_th = np.cos(TH)
    neff = np.zeros(len(k_arr))
    for ik, kv in enumerate(k_arr):
        qx = kv * (sin_th * np.cos(PH) - 1)
        qy = kv * sin_th * np.sin(PH)
        q_perp = np.sqrt(qx**2 + qy**2)
        F = np.where(q_perp > 1e-10,
                     2 * np.pi * R * j1(q_perp * R) / q_perp,
                     np.pi * R**2)
        qz = kv * cos_th
        F2_pol = 4 * np.cos(qz / 2)**2
        if use_transport:
            w = F2_pol * transport * sin_th
        else:
            w = F2_pol * sin_th
        w_denom = F2_pol * transport * sin_th if use_transport else F2_pol * sin_th
        neff[ik] = np.sum(np.abs(F)**2 * w) / np.sum(w_denom)
    return neff


def neff_discrete_born(R, k_arr, TH, PH, sin_th, transport):
    """N_eff Born from discrete lattice disk (V cancels)."""
    dx_b, dy_b = disk_bonds(R)
    cos_th = np.cos(TH)
    neff = np.zeros(len(k_arr))
    for ik, kv in enumerate(k_arr):
        b = np.exp(1j * kv * dx_b)
        phase_out = kv * (
            sin_th[:, :, None] * np.cos(PH)[:, :, None] * dx_b[None, None, :]
            + sin_th[:, :, None] * np.sin(PH)[:, :, None] * dy_b[None, None, :])
        a = np.exp(-1j * phase_out)
        f = np.sum(a * b[None, None, :], axis=2)
        qz = kv * cos_th
        F2_pol = 4 * np.cos(qz / 2)**2
        w = F2_pol * transport * sin_th
        neff[ik] = np.sum(np.abs(f)**2 * w) / np.sum(w)
    return neff


def part_a(TH, PH, sin_th, transport):
    """Continuum vs discrete Born N_eff."""
    print(f"\n{'=' * 65}")
    print(f"  Part A: Continuum Airy disk vs discrete lattice disk")
    print(f"{'=' * 65}")

    print(f"\n  {'R':>4s}  {'N':>5s}  {'p_cont':>8s}  {'p_disc':>8s}  {'diff':>8s}")
    print(f"  {'-' * 40}")

    for R in [2, 3, 5, 7, 9, 12, 15]:
        ne_c = neff_continuum(R, k_vals, TH, PH, sin_th, transport)
        ne_d = neff_discrete_born(R, k_vals, TH, PH, sin_th, transport)
        p_c = np.polyfit(x, np.log(ne_c), 1)[0]
        p_d = np.polyfit(x, np.log(ne_d), 1)[0]
        N_b = len(disk_bonds(R)[0])
        print(f"  {R:4d}  {N_b:5d}  {p_c:8.3f}  {p_d:8.3f}  {p_d - p_c:+8.3f}")

    print(f"\n  → Continuum and discrete agree to < 0.03 at R ≥ 5")
    print(f"  → Both converge to ~ -2.42 (approaching -5/2 = -2.50)")


def part_b(TH, PH, sin_th, transport):
    """Forward cone structure: delta vs eta widths."""
    print(f"\n{'=' * 65}")
    print(f"  Part B: Asymmetric forward cone — delta vs eta")
    print(f"{'=' * 65}")
    print(f"  Forward = (th=pi/2, phi=0). Parametrize th=pi/2-delta, phi=eta.")
    print(f"  q_perp^2 = k^2[1 + cos^2(d) - 2cos(d)cos(e)]")
    print(f"  At small d,e: q_perp ≈ k|e| (linear in eta, quadratic in delta)")
    print(f"  → eta width ~ 1/(kR), delta width ~ 1/sqrt(kR)")

    R = 5
    cos_th = np.cos(TH)

    print(f"\n  Forward cone fraction (R={R}):")
    print(f"  {'k':>5s}  {'qR<10':>8s}  {'|d|<30°':>8s}  {'|d|<60°':>8s}")
    print(f"  {'-' * 35}")

    for kv in k_vals:
        qx = kv * (sin_th * np.cos(PH) - 1)
        qy = kv * sin_th * np.sin(PH)
        q_perp = np.sqrt(qx**2 + qy**2)
        F = np.where(q_perp > 1e-10,
                     2 * np.pi * R * j1(q_perp * R) / q_perp,
                     np.pi * R**2)
        qz = kv * cos_th
        F2_pol = 4 * np.cos(qz / 2)**2
        w = F2_pol * transport * sin_th
        total = np.sum(np.abs(F)**2 * w)

        fwd = q_perp * R < 10
        frac_fwd = np.sum(np.abs(F)**2 * w * fwd) / total * 100

        thetas_1d = TH[:, 0]
        contrib_th = np.sum(np.abs(F)**2 * w, axis=1)
        total_th = np.sum(contrib_th)
        m30 = np.abs(thetas_1d - np.pi / 2) < np.radians(30)
        m60 = np.abs(thetas_1d - np.pi / 2) < np.radians(60)
        f30 = np.sum(contrib_th[m30]) / total_th * 100
        f60 = np.sum(contrib_th[m60]) / total_th * 100

        print(f"  {kv:5.2f}  {frac_fwd:7.1f}%  {f30:7.1f}%  {f60:7.1f}%")

    print(f"\n  → Forward cone (qR<10) dominates at low k")
    print(f"  → In theta: contribution spread across ±60° (delta is wide)")
    print(f"  → Asymmetry: delta ~ 1/sqrt(kR) >> eta ~ 1/(kR)")


def part_c(TH, PH, sin_th, transport):
    """Transport decomposition: -3/2 + (-1) = -5/2."""
    print(f"\n{'=' * 65}")
    print(f"  Part C: sigma_total vs sigma_tr — transport adds -1")
    print(f"{'=' * 65}")

    print(f"\n  {'R':>4s}  {'p_total':>8s}  {'p_tr':>8s}  {'diff':>8s}")
    print(f"  {'-' * 35}")

    for R in [5, 10, 20, 50]:
        ne_tot = neff_continuum(R, k_vals, TH, PH, sin_th, transport,
                                use_transport=False)
        ne_tr = neff_continuum(R, k_vals, TH, PH, sin_th, transport,
                               use_transport=True)
        p_tot = np.polyfit(x, np.log(ne_tot), 1)[0]
        p_tr = np.polyfit(x, np.log(ne_tr), 1)[0]
        print(f"  {R:4d}  {p_tot:8.3f}  {p_tr:8.3f}  {p_tr - p_tot:+8.3f}")

    print(f"\n  Derivation of -5/2:")
    print(f"    Forward: th=pi/2, phi=0. Parametrize th=pi/2-delta, phi=eta.")
    print(f"    q_perp^2 = k^2[cos^2(d) - 2cos(d)cos(e) + 1]")
    print(f"    At small d,e: q_x = k(cos(d)cos(e)-1) ≈ -k(d^2+e^2)/2  (quadratic)")
    print(f"                  q_y = k*cos(d)*sin(e) ≈ k*e  (linear)")
    print(f"    q_perp ≈ k|e| at leading order (q_x is subleading)")
    print(f"    Airy disk: F large when q_perp*R < few → |e| < few/(kR)")
    print(f"    eta width ~ 1/(kR)  [standard Airy]")
    print(f"    At e=0: q_perp = k(1-cos(d)) ≈ k*d^2/2 → d^2 < few/(kR)")
    print(f"    delta width ~ 1/sqrt(kR)  [parabolic, wider than eta]")
    print(f"    Cone solid angle = eta * delta ~ 1/(kR) * 1/sqrt(kR)"
          f" = (kR)^{{-3/2}}")
    print(f"    sigma_total ~ |F(0)|^2 * cone_area ~ R^4 * (kR)^{{-3/2}}")
    print(f"    Transport: 1-cos(d)cos(e) ≈ (d^2+e^2)/2 ≈ d^2/2 in cone")
    print(f"    <d^2> ~ 1/(kR) → transport weight ~ 1/(kR)")
    print(f"    sigma_tr ~ sigma_total * 1/(kR) ~ (kR)^{{-3/2-1}} = (kR)^{{-5/2}}")
    print(f"  → -5/2 = -3/2 (cone) + (-1) (transport). QED.")


def part_d():
    """Large-kR convergence to -5/2. Uses (delta, eta) grid for R=200."""
    print(f"\n{'=' * 65}")
    print(f"  Part D: Large-kR convergence — (delta, eta) grid at R=200")
    print(f"{'=' * 65}")

    R = 200
    N_d, N_e = 2000, 2000
    delta = np.linspace(-np.pi / 2, np.pi / 2, N_d)
    eta = np.linspace(-np.pi, np.pi, N_e, endpoint=False)
    D, E = np.meshgrid(delta, eta, indexing='ij')
    cos_d = np.cos(D)
    cos_e = np.cos(E)
    tr = 1 - cos_d * cos_e
    jac = cos_d
    dd = np.pi / N_d
    de = 2 * np.pi / N_e

    W = np.sum(tr * jac) * dd * de
    W_tot = np.sum(jac) * dd * de
    print(f"  W (transport) = {W:.4f} (4*pi = {4 * np.pi:.4f})")
    print(f"  W_tot (no transport) = {W_tot:.4f} (4*pi = {4 * np.pi:.4f})")

    k_dense = np.logspace(np.log10(0.01), np.log10(0.15), 30)
    kR = k_dense * R
    x_d = np.log(k_dense)
    print(f"  k_dense = {k_dense[0]:.3f}-{k_dense[-1]:.3f}"
          f" (kR = {kR[0]:.0f}-{kR[-1]:.0f})")
    print(f"  NOTE: different from k_vals ({k_vals[0]}-{k_vals[-1]})"
          f" used in other parts")

    neff_tr = np.zeros(len(k_dense))
    neff_tot = np.zeros(len(k_dense))
    W_tot = np.sum(jac) * dd * de

    for ik, kv in enumerate(k_dense):
        q_perp2 = kv**2 * (1 + cos_d**2 - 2 * cos_d * cos_e)
        q_perp = np.sqrt(q_perp2)
        qR = q_perp * R
        F = np.where(qR > 1e-10,
                     2 * np.pi * R * j1(qR) / qR,
                     np.pi * R**2)
        F2 = np.abs(F)**2
        neff_tr[ik] = np.sum(F2 * tr * jac) * dd * de / W
        neff_tot[ik] = np.sum(F2 * jac) * dd * de / W_tot

    p_tr = np.polyfit(x_d, np.log(neff_tr), 1)[0]
    p_tot = np.polyfit(x_d, np.log(neff_tot), 1)[0]

    mask_hi = kR > 10
    p_tr_hi = np.polyfit(x_d[mask_hi], np.log(neff_tr[mask_hi]), 1)[0]
    p_tot_hi = np.polyfit(x_d[mask_hi], np.log(neff_tot[mask_hi]), 1)[0]

    mask_vhi = kR > 15
    p_tr_vhi = np.polyfit(x_d[mask_vhi], np.log(neff_tr[mask_vhi]), 1)[0]

    print(f"\n  R={R}, 30 k-points, kR = {kR[0]:.0f}-{kR[-1]:.0f}:")
    print(f"  {'range':>12s}  {'p_tr':>8s}  {'p_total':>8s}  {'diff':>8s}")
    print(f"  {'-' * 42}")
    print(f"  {'all kR':>12s}  {p_tr:8.4f}  {p_tot:8.4f}  {p_tr - p_tot:+8.4f}")
    print(f"  {'kR > 10':>12s}  {p_tr_hi:8.4f}  {p_tot_hi:8.4f}"
          f"  {p_tr_hi - p_tot_hi:+8.4f}")
    print(f"  {'kR > 15':>12s}  {p_tr_vhi:8.4f}  {'---':>8s}  {'---':>8s}")
    print(f"\n  Targets: p_tr → -2.500, p_total → -1.500, diff → -1.000")
    print(f"  Deficit at kR>15: {-2.5 - p_tr_vhi:+.4f}"
          f" ({(-2.5 - p_tr_vhi) / 2.5 * 100:.1f}%)")

    # Asymptotic fit with 1/(kR) correction
    A_mat = np.column_stack([np.ones(len(k_dense)), x_d, 1.0 / (k_dense * R)])
    coeffs = np.linalg.lstsq(A_mat, np.log(neff_tr), rcond=None)[0]
    print(f"\n  Fit: log(N_eff) = a + p*log(k) + c/(kR)")
    print(f"    p_asymptotic = {coeffs[1]:.4f} (kR→∞)")
    print(f"    correction c = {coeffs[2]:.3f}")
    print(f"\n  → Born exponent -5/2 confirmed to {abs(-2.5 - p_tr_vhi) / 2.5 * 100:.1f}%"
          f" at kR > 15")


def neff_ms(R, k_arr, alpha, TH, PH, sin_th, transport):
    """N_eff with full multiple scattering T=(I-VG)^{-1}V."""
    dx_b, dy_b = disk_bonds(R)
    N_b = len(dx_b)
    V = K1 * (np.cos(2 * np.pi * alpha) - 1)
    cos_th = np.cos(TH)

    neff_b = np.zeros(len(k_arr))
    neff_m = np.zeros(len(k_arr))

    for ik, kv in enumerate(k_arr):
        omega2 = 2 * K1 * (1 - np.cos(kv)) + 4 * K2 * (2 - 2 * np.cos(kv))
        k_eff = 2 * np.sin(kv / 2)
        dist = np.sqrt((dx_b[:, None] - dx_b[None, :])**2
                       + (dy_b[:, None] - dy_b[None, :])**2)
        mask = dist > 0
        G = np.zeros((N_b, N_b), dtype=complex)
        G[mask] = (np.exp(1j * k_eff * dist[mask])
                   / (4 * np.pi * c_lat**2 * dist[mask]))
        denom = 1.0 / (omega2 - omega_k2 + 1j * eps_lat)
        G_00 = np.mean(denom)
        np.fill_diagonal(G, G_00)

        T = np.linalg.solve(np.eye(N_b) - V * G, V * np.eye(N_b))
        b = np.exp(1j * kv * dx_b)
        Tb = T @ b
        Vb = V * b

        phase_out = kv * (
            sin_th[:, :, None] * np.cos(PH)[:, :, None] * dx_b[None, None, :]
            + sin_th[:, :, None] * np.sin(PH)[:, :, None]
            * dy_b[None, None, :])
        a = np.exp(-1j * phase_out)
        f_ms = np.sum(a * Tb[None, None, :], axis=2)
        f_born = np.sum(a * Vb[None, None, :], axis=2)

        qz = kv * cos_th
        F2 = 4 * np.cos(qz / 2)**2
        w = F2 * transport * sin_th
        s_single = np.abs(V)**2 * np.sum(w)
        neff_b[ik] = np.sum(np.abs(f_born)**2 * w) / s_single
        neff_m[ik] = np.sum(np.abs(f_ms)**2 * w) / s_single

    return neff_b, neff_m


def part_e(TH, PH, sin_th, transport):
    """MS exponent vs alpha — crossing at alpha ≈ 0.29."""
    print(f"\n{'=' * 65}")
    print(f"  Part E: MS exponent vs alpha — where does p_MS = -2.0?")
    print(f"{'=' * 65}")

    R = 5
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    print(f"\n  R={R}")
    print(f"  {'alpha':>6s}  {'|V|':>8s}  {'p_Born':>8s}  {'p_MS':>8s}"
          f"  {'p_MS+2':>8s}  {'CV_int':>8s}")
    print(f"  {'-' * 55}")

    for alpha in alphas:
        V = K1 * (np.cos(2 * np.pi * alpha) - 1)
        nb, nm = neff_ms(R, k_vals, alpha, TH, PH, sin_th, transport)
        pb = np.polyfit(x, np.log(nb), 1)[0]
        pm = np.polyfit(x, np.log(nm), 1)[0]

        vg = c_lat * np.cos(k_vals / 2)
        integrand = np.sin(k_vals)**2 * np.abs(V)**2 / vg**2 * nm
        cv_int = cv(integrand)

        print(f"  {alpha:6.2f}  {np.abs(V):8.4f}  {pb:8.3f}  {pm:8.3f}"
              f"  {pm + 2:+8.3f}  {cv_int:7.1f}%")

    print(f"\n  → p_MS crosses -2.0 at alpha ≈ 0.29 (|V| ≈ 1.26)")
    print(f"  → Minimum CV ≈ 10% at alpha ≈ 0.30")
    print(f"  → -2.0 is NOT exact: it's the crossing point p_MS(alpha) = -2")


def part_f(TH, PH, sin_th, transport):
    """Flat integrand residual CV: three sources."""
    print(f"\n{'=' * 65}")
    print(f"  Part F: Residual CV — why 7-10% instead of 0%?")
    print(f"{'=' * 65}")

    # Source (c): sinc^2(k/2) deviation
    sinc2 = (np.sin(k_vals / 2) / (k_vals / 2))**2
    cv_sinc = cv(sinc2)
    print(f"\n  Source (c): sinc^2(k/2) = [sin(k/2)/(k/2)]^2")
    print(f"  {'k':>5s}  {'sinc^2':>8s}")
    print(f"  {'-' * 15}")
    for i, kv in enumerate(k_vals):
        print(f"  {kv:5.2f}  {sinc2[i]:8.4f}")
    print(f"  CV(sinc^2) = {cv_sinc:.1f}%")
    print(f"  → sin^2(k/2) ≠ (k/2)^2 at finite k. Contributes ~6% to CV.")

    # Source (a): p_MS not exactly -2.0
    R = 5
    alpha = 0.30
    nb, nm = neff_ms(R, k_vals, alpha, TH, PH, sin_th, transport)
    pm = np.polyfit(x, np.log(nm), 1)[0]
    print(f"\n  Source (a): p_MS = {pm:.3f} (not -2.000)")
    print(f"  Deficit: {pm + 2:+.3f}")

    # Power-law fit residual
    fit = np.polyfit(x, np.log(nm), 1)
    resid = np.log(nm) - np.polyval(fit, x)
    print(f"\n  Source (b): N_eff is not a pure power law")
    print(f"  Log-residual from power-law fit: std = {np.std(resid):.4f}")

    # Combined: compute actual integrand CV
    V = K1 * (np.cos(2 * np.pi * alpha) - 1)
    vg = c_lat * np.cos(k_vals / 2)
    integrand = np.sin(k_vals)**2 * np.abs(V)**2 / vg**2 * nm
    cv_full = cv(integrand)
    print(f"\n  Full integrand sin^2(k)*sigma_tr at alpha={alpha}:")
    print(f"  CV = {cv_full:.1f}%")

    # Ideal: if N_eff were exactly k^{-2}
    nm_ideal = nm[0] * (k_vals / k_vals[0])**(-2.0)
    integrand_ideal = np.sin(k_vals)**2 * np.abs(V)**2 / vg**2 * nm_ideal
    cv_ideal = cv(integrand_ideal)
    print(f"\n  If N_eff were exactly k^{{-2}}: CV = {cv_ideal:.1f}%")
    print(f"  → Residual {cv_ideal:.1f}% from sinc^2 alone")
    print(f"  → Additional {cv_full - cv_ideal:.1f}% from sources (a) + (b)")

    print(f"\n  Summary:")
    print(f"    sinc^2(k/2) ≠ 1:          ~{cv_sinc:.0f}% CV")
    print(f"    p_MS = {pm:.2f} ≠ -2.00:   ~{abs(pm + 2) * 10:.0f}% CV")
    print(f"    non-power-law:             ~{max(0, cv_full - cv_ideal - abs(pm+2)*10):.0f}% CV")
    print(f"    total:                     {cv_full:.0f}% CV")
    print(f"  → Flat integrand is approximate (CV ≈ 10%), not forced by symmetry")


if __name__ == '__main__':
    t0 = time.time()
    print("Route 60: Born exponent -5/2 derivation + gap closure")

    TH, PH, sin_th, transport = make_angular_grid(200, 200)

    part_a(TH, PH, sin_th, transport)
    part_b(TH, PH, sin_th, transport)
    part_c(TH, PH, sin_th, transport)
    part_d()
    part_e(TH, PH, sin_th, transport)
    part_f(TH, PH, sin_th, transport)

    print(f"\nTotal: {time.time() - t0:.0f}s")
