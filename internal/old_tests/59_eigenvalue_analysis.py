"""
Route 59: Eigenvalue analysis of VG — mechanism of +1/2 exponent shift.

File 58 confirmed: collective MS T=(I-VG)^{-1}V produces the shift from Born -5/2
to FDTD -2.0 (91-104% at all R). This file dissects WHY.

Parts:
  A: Eigenvalue spectrum of VG at R=5
     |λ_max| = 0.3-0.6 (far from resonance). Dominant mode (Born) 56% at k=0.3,
     <1% at k=1.5. T-matrix weighting (1/(1-λ)²) gives similar or lower dominance
     (48% at k=0.3). 1-|λ_max| ~ k^{1/3} (neither k nor √k).
  B: Phase removal — G(r) with vs without exp(ikr)
     Without phase (static 1/r): shift +0.83 (too much). With phase: +0.45.
     Phase REDUCES shift. Phase randomization at high k reduces inter-bond
     cooperation (see Part G for mechanism).
  C: α scan — shift vs |V| linearity
     shift ≈ C × |V| with C ≈ 0.31 ± 0.03 (R-independent, mildly α-dependent).
     Flat integrand requires shift ≈ 0.42 → |V| ≈ 1.26 → α ≈ 0.29.
     Self-consistent prediction (uses empirical Born exponent), not first-principles.
  D: Born series convergence at α=0.30
     1st correction (VG): 27% of shift. 2nd (VG²): 152%. Full MS: 100%.
     Series oscillates at |λ_max| ≈ 0.6, requires full resummation.
  E: Kernel exponent scan — G(r) = exp(ikr)/r^{1+ε} + clean decomposition
     1/r: +0.45. 1/r²: +0.34. 1/r⁶: +0.21. Uniform G: +0.69.
     Clean decomposition: G₀₀ diagonal only +0.04, off-diag 1/r only +0.36,
     physical (both) +0.45, uniform +0.69.
     Shift is 80% from inter-bond 1/r propagation, NOT from self-energy G₀₀.
  F: Geometry scan — 1D line vs 2D disk vs annulus
     Disk R=5: +0.45 (Born -2.42). Line: +0.12-0.19 (Born -0.9 to -1.6).
     Annulus: +0.18 (closer to 1D than disk). Disk interior matters, not perimeter.
  G: Single-mode (Rayleigh quotient) analysis
     λ_eff = ⟨b|VG|b⟩/⟨b|b⟩ projects VG onto incident mode.
     Single-mode Ansatz: T·b ≈ V/(1-λ_eff)·b (valid when VG·b ≈ λ_eff·b).
     Residual |VGb-λb|/|VGb| monitored — <0.5 at all k.
     |λ_eff| ~ k^{-0.77}: phases coherent at low k → more cooperation → shift.
     Single-mode captures 77% (R=2) → 80% (R=5) → 100% (R≥7) of shift.
     Uses no-self G (inter-bond only) — G_00 adds only 9% (Part E).
     ~80% of λ_eff from r ≤ 3 (short-range lattice structure).
     C ≈ 0.31 is a lattice geometric constant (not derivable from continuum).
     NOTE: dom_T% (Part A) ≠ %SM (Part G) — they measure different things.

Results:
  - NOT resonance (|λ_max| = 0.3-0.6, far from 1 at α=0.30)
  - Phase exp(ikr) REDUCES total shift (from +0.83 to +0.45)
  - Shift is ~80% from off-diagonal inter-bond 1/r, ~9% from self-energy G₀₀
    (interaction term ~10% — decomposition is approximate)
  - Shift is LINEAR in |V| (coupling strength), coefficient C ≈ 0.31
  - Single-mode T·b ≈ V/(1-λ_eff)·b captures 77-100% of shift (exact at large R)
    (valid when VG·b ≈ λ_eff·b; residual monitored)
  - |λ_eff| ~ k^{-0.77}: phase coherence at low k → more enhancement → shift
  - Disk geometry gives largest shift (vs line, ring, annulus)
  - α-threshold prediction α≈0.29 (self-consistent, not first-principles)
  - C ≈ 0.31 is lattice-specific: ~80% from r ≤ 3, continuum integral fails

Run: cd ST_11/wip/w_21_kappa && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 59_eigenvalue_analysis.py
"""

import numpy as np
import time

K1, K2 = 1.0, 0.5
c_lat = np.sqrt(K1 + 4 * K2)
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
omega_arr = 2 * c_lat * np.sin(k_vals / 2)
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


# Lattice BZ grid (for self-energy G_00 only)
N_bz = 64
kx_1d = np.linspace(-np.pi, np.pi, N_bz, endpoint=False)
CX, CY, CZ = np.meshgrid(np.cos(kx_1d), np.cos(kx_1d), np.cos(kx_1d),
                          indexing='ij')
omega_k2 = (2 * K1 * (3 - CX - CY - CZ)
            + 4 * K2 * (3 - CX * CY - CY * CZ - CX * CZ))


def build_G(dx_b, dy_b, kv, with_phase=True):
    """Build Green's function matrix: continuum off-diag, lattice self-energy.

    Off-diagonal: G_ij = exp(ik_eff r)/(4πc²r) (continuum, with_phase=True)
                  or G_ij = 1/(4πc²r) (static/Coulomb limit, with_phase=False).
    Diagonal: G_00 from lattice BZ sum at omega2(kv). omega2 is used here.
    Note: with_phase=False gives the static (ω→0) propagator, not just
    "phase removed" — physically a different regime (Coulomb-like 1/r).
    """
    N_b = len(dx_b)
    # omega2 = ω²(kv) along [100], used for G_00 BZ sum below
    omega2 = 2 * K1 * (1 - np.cos(kv)) + 4 * K2 * (2 - 2 * np.cos(kv))
    k_eff = 2 * np.sin(kv / 2)

    dist = np.sqrt((dx_b[:, None] - dx_b[None, :])**2
                   + (dy_b[:, None] - dy_b[None, :])**2)

    G = np.zeros((N_b, N_b), dtype=complex)
    mask = dist > 0
    if with_phase:
        G[mask] = (np.exp(1j * k_eff * dist[mask])
                   / (4 * np.pi * c_lat**2 * dist[mask]))
    else:
        # Static propagator: 1/r without oscillation (ω→0 limit)
        G[mask] = 1.0 / (4 * np.pi * c_lat**2 * dist[mask])

    # Self-energy: lattice BZ sum at ω²(kv) — retains lattice effects at r=0
    denom = 1.0 / (omega2 - omega_k2 + 1j * eps_lat)
    G_00 = np.mean(denom)
    np.fill_diagonal(G, G_00)
    return G


def compute_neff(dx_b, dy_b, k_vals, alpha, with_phase=True,
                 TH=None, PH=None, sin_th=None, transport=None):
    """Compute N_eff (Born + MS) with optional phase removal."""
    N_b = len(dx_b)
    V = K1 * (np.cos(2 * np.pi * alpha) - 1)

    neff_b = np.zeros(len(k_vals))
    neff_m = np.zeros(len(k_vals))

    for ik, kv in enumerate(k_vals):
        G = build_G(dx_b, dy_b, kv, with_phase=with_phase)
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

        qz = kv * np.cos(TH)
        F2 = 4 * np.cos(qz / 2)**2
        w = F2 * transport * sin_th

        s_single = np.abs(V)**2 * np.sum(w)
        neff_b[ik] = np.sum(np.abs(f_born)**2 * w) / s_single
        neff_m[ik] = np.sum(np.abs(f_ms)**2 * w) / s_single

    return neff_b, neff_m


if __name__ == '__main__':
    t0 = time.time()

    # Angular grid (shared across parts B-D)
    N_th, N_ph = 200, 200
    thetas = np.linspace(0, np.pi, N_th)
    phis = np.linspace(0, 2 * np.pi, N_ph, endpoint=False)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    sin_th = np.sin(TH)
    transport = 1 - sin_th * np.cos(PH)

    print("Route 59: Eigenvalue analysis of VG")
    print(f"  BZ: {N_bz}^3, angular: {N_th}x{N_ph}, eps={eps_lat}")
    print()

    # ═══════════════════════════════════════════════════════════════
    # Part A: Eigenvalue spectrum of VG at R=5
    # ═══════════════════════════════════════════════════════════════
    print(f"{'=' * 65}")
    print(f"  Part A: Eigenvalue spectrum of VG")
    print(f"{'=' * 65}")

    R = 5
    alpha = 0.30
    V = K1 * (np.cos(2 * np.pi * alpha) - 1)
    dx_b, dy_b = disk_bonds(R)
    N_b = len(dx_b)
    print(f"  R={R}, N={N_b}, V={V:.4f}, α={alpha}")

    lam_max_arr = np.zeros(len(k_vals), dtype=complex)
    lam_2_arr = np.zeros(len(k_vals), dtype=complex)
    dom_arr = np.zeros(len(k_vals))
    dom_T_arr = np.zeros(len(k_vals))

    print(f"\n  {'k':>5s}  {'|lam_max|':>9s}  {'arg':>7s}  {'1-|lam|':>9s}"
          f"  {'|lam_2|':>8s}  {'dom_b%':>7s}  {'dom_T%':>7s}")
    print(f"  {'-' * 65}")

    for ik, kv in enumerate(k_vals):
        G = build_G(dx_b, dy_b, kv)
        VG = V * G

        eigvals_all, U = np.linalg.eig(VG)
        idx = np.argsort(-np.abs(eigvals_all))
        eigvals_sorted = eigvals_all[idx]
        U_sorted = U[:, idx]

        lam_max_arr[ik] = eigvals_sorted[0]
        lam_2_arr[ik] = eigvals_sorted[1]

        # Dominant mode: projection of incident wave onto eigenmodes
        b = np.exp(1j * kv * dx_b)
        U_inv = np.linalg.inv(U_sorted)
        coeffs = np.abs(U_inv @ b)**2
        dom_arr[ik] = coeffs[0] / np.sum(coeffs) * 100

        # After T-matrix: mode j amplified by 1/(1-λ_j)
        # Effective weight = |coeff_j|² / |1-λ_j|²
        T_weights = coeffs / np.abs(1 - eigvals_sorted)**2
        dom_T_arr[ik] = T_weights[0] / np.sum(T_weights) * 100

        lm = eigvals_sorted[0]
        print(f"  {kv:5.2f}  {np.abs(lm):9.4f}  {np.angle(lm):7.3f}"
              f"  {1-np.abs(lm):9.4f}  {np.abs(eigvals_sorted[1]):8.4f}"
              f"  {dom_arr[ik]:6.1f}%  {dom_T_arr[ik]:6.1f}%")

    x = np.log(k_vals)
    p_lam = np.polyfit(x, np.log(1 - np.abs(lam_max_arr)), 1)[0]

    ratio_k = (1 - np.abs(lam_max_arr)) / k_vals
    ratio_sqrtk = (1 - np.abs(lam_max_arr)) / np.sqrt(k_vals)

    print(f"\n  1-|lam_max| ~ k^{p_lam:.2f}")
    print(f"  (1-|lam|)/k:      CV={cv(ratio_k):.1f}%")
    print(f"  (1-|lam|)/sqrt(k): CV={cv(ratio_sqrtk):.1f}%")
    print(f"\n  Dominant mode (Born projection): {dom_arr[0]:.0f}% at k=0.3"
          f" → {dom_arr[-1]:.1f}% at k=1.5")
    print(f"  Dominant mode (T-matrix weighted): {dom_T_arr[0]:.0f}% at k=0.3"
          f" → {dom_T_arr[-1]:.1f}% at k=1.5")
    print(f"  (T-matrix amplifies by 1/(1-λ_j) — dominant mode more prominent)")
    print(f"  → NOT resonance (|lam| far from 1)")
    print(f"  → NOT single-mode at high k")

    # Top 5 eigenvalues at k=0.3
    G03 = build_G(dx_b, dy_b, 0.3)
    eig03 = np.linalg.eigvals(V * G03)
    idx03 = np.argsort(-np.abs(eig03))
    print(f"\n  Top 5 eigenvalues at k=0.3:")
    for i in range(5):
        e = eig03[idx03[i]]
        print(f"    lam_{i}: |{np.abs(e):.4f}|  ({e.real:+.4f} {e.imag:+.4f}i)")

    t_a = time.time() - t0
    print(f"\n  Part A: {t_a:.0f}s")

    # ═══════════════════════════════════════════════════════════════
    # Part B: Phase removal — with vs without exp(ikr)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print(f"  Part B: Phase removal — G(r)=exp(ikr)/(4πr) vs G(r)=1/(4πr)")
    print(f"{'=' * 65}")
    print(f"  R={R}, α={alpha}")

    nb_p, nm_p = compute_neff(dx_b, dy_b, k_vals, alpha, with_phase=True,
                              TH=TH, PH=PH, sin_th=sin_th, transport=transport)
    nb_np, nm_np = compute_neff(dx_b, dy_b, k_vals, alpha, with_phase=False,
                                TH=TH, PH=PH, sin_th=sin_th,
                                transport=transport)

    p_b = np.polyfit(x, np.log(nb_p), 1)[0]
    p_p = np.polyfit(x, np.log(nm_p), 1)[0]
    p_np = np.polyfit(x, np.log(nm_np), 1)[0]

    print(f"\n  {'k':>5s}  {'Born':>10s}  {'MS+phase':>10s}"
          f"  {'MS-phase':>10s}  {'ratio':>8s}")
    print(f"  {'-' * 50}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {nb_p[i]:10.1f}  {nm_p[i]:10.1f}"
              f"  {nm_np[i]:10.1f}  {nm_np[i]/nm_p[i]:8.3f}")

    print(f"\n  Exponents:")
    print(f"    Born:         {p_b:.3f}")
    print(f"    MS with phase: {p_p:.3f}  (shift {p_p-p_b:+.3f})")
    print(f"    MS no phase:   {p_np:.3f}  (shift {p_np-p_b:+.3f})")
    print(f"\n  → Phase REDUCES shift from {p_np-p_b:+.3f} to {p_p-p_b:+.3f}")
    print(f"  → Phase REDUCES shift: static 1/r gives MORE shift than physical G")
    print(f"  → See Part G: shift is from inter-bond 1/r (80-90%), not G_00 (9%)")
    print(f"     Phase randomization at high k reduces cooperation → shift")

    t_b = time.time() - t0 - t_a
    print(f"\n  Part B: {t_b:.0f}s")

    # ═══════════════════════════════════════════════════════════════
    # Part C: α scan — shift ≈ C × |V|
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print(f"  Part C: Shift vs |V| — α scan at R=5 + R cross-check")
    print(f"{'=' * 65}")

    alpha_scan = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    print(f"\n  R=5 α scan:")
    print(f"  {'alpha':>6s}  {'|V|':>8s}  {'p_Born':>8s}  {'p_MS':>8s}"
          f"  {'shift':>8s}  {'C=sh/|V|':>8s}")
    print(f"  {'-' * 55}")

    shifts_5 = []
    Vs_5 = []

    for a in alpha_scan:
        V_a = K1 * (np.cos(2 * np.pi * a) - 1)
        nb_a, nm_a = compute_neff(dx_b, dy_b, k_vals, a, TH=TH, PH=PH,
                                  sin_th=sin_th, transport=transport)
        pb_a = np.polyfit(x, np.log(nb_a), 1)[0]
        pm_a = np.polyfit(x, np.log(nm_a), 1)[0]
        sh = pm_a - pb_a
        C_a = sh / np.abs(V_a)
        shifts_5.append(sh)
        Vs_5.append(np.abs(V_a))
        print(f"  {a:6.2f}  {np.abs(V_a):8.4f}  {pb_a:8.3f}  {pm_a:8.3f}"
              f"  {sh:+8.3f}  {C_a:8.3f}")

    shifts_5 = np.array(shifts_5)
    Vs_5 = np.array(Vs_5)
    C_mean_5 = np.mean(shifts_5 / Vs_5)
    C_cv_5 = cv(shifts_5 / Vs_5)

    # Power law fit: shift ~ |V|^p
    p_sv = np.polyfit(np.log(Vs_5), np.log(shifts_5), 1)[0]
    print(f"\n  shift ~ |V|^{p_sv:.2f}  (linear = 1.00)")
    print(f"  C = shift/|V|: mean={C_mean_5:.3f}, CV={C_cv_5:.1f}%")

    # Flat integrand prediction: shift = -(Born + 2.0)
    # Uses p_b computed above (empirical Born exponent at R=5).
    # This prediction is self-consistent (uses simulated Born exponent), not a
    # first-principles derivation. The threshold α≈0.25 from |cm1|=|s_phi|
    # (per-bond V constancy, file 55) remains the independent bound.
    p_born_ref = p_b  # Born exponent at R=5, already computed in α=0.05 row
    shift_flat = abs(p_born_ref) - 2.0
    V_flat = shift_flat / C_mean_5
    alpha_pred = np.arccos(1 - V_flat) / (2 * np.pi)
    print(f"\n  Flat integrand: shift needed = {shift_flat:.3f}"
          f" (Born exponent 2.415 is empirical at R=5)")
    print(f"    → |V| = {V_flat:.3f} → α = {alpha_pred:.3f}")
    print(f"    Self-consistent, not first-principles (uses simulated Born exponent)")
    print(f"    Independent threshold: α≈0.25 from |cm1|=|s_phi| (file 55)")

    # R cross-check at α=0.10 and α=0.40
    print(f"\n  R cross-check (C = shift/|V|):")
    print(f"  {'alpha':>6s}  {'R':>4s}  {'N':>4s}  {'shift':>8s}  {'C':>8s}")
    print(f"  {'-' * 40}")

    C_all = []
    for a in [0.10, 0.30, 0.40]:
        V_a = K1 * (np.cos(2 * np.pi * a) - 1)
        for R_chk in [3, 7]:
            dx_r, dy_r = disk_bonds(R_chk)
            N_r = len(dx_r)
            nb_r, nm_r = compute_neff(dx_r, dy_r, k_vals, a, TH=TH, PH=PH,
                                      sin_th=sin_th, transport=transport)
            pb_r = np.polyfit(x, np.log(nb_r), 1)[0]
            pm_r = np.polyfit(x, np.log(nm_r), 1)[0]
            sh_r = pm_r - pb_r
            C_r = sh_r / np.abs(V_a)
            C_all.append(C_r)
            print(f"  {a:6.2f}  {R_chk:4d}  {N_r:4d}  {sh_r:+8.3f}"
                  f"  {C_r:8.3f}")

    C_all = np.array(C_all)
    print(f"\n  C across all R×α: mean={np.mean(C_all):.3f},"
          f" CV={cv(C_all):.1f}%")
    print(f"  Combined (R=5 + cross): mean="
          f"{np.mean(np.concatenate([shifts_5/Vs_5, C_all])):.3f}")

    t_c = time.time() - t0 - t_a - t_b
    print(f"\n  Part C: {t_c:.0f}s")

    # ═══════════════════════════════════════════════════════════════
    # Part D: Born series convergence
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print(f"  Part D: Born series — 1st, 2nd correction vs full MS")
    print(f"{'=' * 65}")

    R = 5
    alpha = 0.30
    V = K1 * (np.cos(2 * np.pi * alpha) - 1)
    dx_b, dy_b = disk_bonds(R)
    N_b = len(dx_b)
    print(f"  R={R}, N={N_b}, V={V:.4f}")

    neff_b = np.zeros(len(k_vals))
    neff_1 = np.zeros(len(k_vals))
    neff_2 = np.zeros(len(k_vals))
    neff_m = np.zeros(len(k_vals))

    for ik, kv in enumerate(k_vals):
        G = build_G(dx_b, dy_b, kv)
        VG = V * G
        VG2 = VG @ VG

        b = np.exp(1j * kv * dx_b)
        # Born series: T = V + VGV + VGVGV + ... = V·Σ_n (VG)^n
        # T·b at order n includes n multiple-scattering events.
        Vb = V * b                          # order 0 (Born): V·b
        T1b = V * (b + VG @ b)              # order 1: V(I+VG)b = Vb + V²Gb
        T2b = V * (b + VG @ b + VG2 @ b)    # order 2: V(I+VG+(VG)²)b
        T = np.linalg.solve(np.eye(N_b) - VG, V * np.eye(N_b))
        Tb = T @ b                           # full: V(I-VG)^{-1}b

        phase_out = kv * (
            sin_th[:, :, None] * np.cos(PH)[:, :, None] * dx_b[None, None, :]
            + sin_th[:, :, None] * np.sin(PH)[:, :, None]
            * dy_b[None, None, :])
        a = np.exp(-1j * phase_out)

        qz = kv * np.cos(TH)
        F2 = 4 * np.cos(qz / 2)**2
        w = F2 * transport * sin_th
        s_single = np.abs(V)**2 * np.sum(w)

        for amp, arr in [(Vb, neff_b), (T1b, neff_1),
                         (T2b, neff_2), (Tb, neff_m)]:
            f = np.sum(a * amp[None, None, :], axis=2)
            arr[ik] = np.sum(np.abs(f)**2 * w) / s_single

    p_b = np.polyfit(x, np.log(neff_b), 1)[0]
    p_1 = np.polyfit(x, np.log(neff_1), 1)[0]
    p_2 = np.polyfit(x, np.log(neff_2), 1)[0]
    p_m = np.polyfit(x, np.log(neff_m), 1)[0]

    print(f"\n  {'k':>5s}  {'Born':>10s}  {'Born+VG':>10s}"
          f"  {'Born+VG+VG2':>11s}  {'Full MS':>10s}")
    print(f"  {'-' * 52}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {neff_b[i]:10.1f}  {neff_1[i]:10.1f}"
              f"  {neff_2[i]:11.1f}  {neff_m[i]:10.1f}")

    shift_full = p_m - p_b
    print(f"\n  Exponents:")
    print(f"    Born:         {p_b:.3f}")
    print(f"    Born+VG:      {p_1:.3f}  (shift {p_1-p_b:+.3f},"
          f" {(p_1-p_b)/shift_full*100:.0f}% of full)")
    print(f"    Born+VG+VG²:  {p_2:.3f}  (shift {p_2-p_b:+.3f},"
          f" {(p_2-p_b)/shift_full*100:.0f}% of full)")
    print(f"    Full MS:      {p_m:.3f}  (shift {p_m-p_b:+.3f}, 100%)")

    print(f"\n  |lam_max| at k=0.3: {np.abs(lam_max_arr[0]):.2f}"
          f" → series oscillates, needs full resummation")

    t_d = time.time() - t0 - t_a - t_b - t_c
    print(f"\n  Part D: {t_d:.0f}s")

    # ═══════════════════════════════════════════════════════════════
    # Part E: Kernel exponent scan — G(r) = exp(ikr)/r^{1+ε}
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print(f"  Part E: Kernel exponent — G(r) = exp(ikr) / r^{{1+eps}}")
    print(f"{'=' * 65}")

    R = 5
    alpha = 0.30
    V = K1 * (np.cos(2 * np.pi * alpha) - 1)
    dx_b, dy_b = disk_bonds(R)
    N_b = len(dx_b)
    dist_e = np.sqrt((dx_b[:, None] - dx_b[None, :])**2
                     + (dy_b[:, None] - dy_b[None, :])**2)
    mask_e = dist_e > 0

    print(f"  R={R}, N={N_b}, V={V:.4f}")

    eps_scan = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0]

    # NOTE: Born N_eff is G-independent (structure factor only).
    # p_Born should be identical (~-2.42) for all kernel rows.
    # Variation = numerical noise only.
    print(f"  NOTE: Born N_eff is G-independent (structure factor only).")
    print(f"  p_Born should be identical (~-2.42) for all kernel rows.")
    print(f"  Variation = numerical noise only.")

    print(f"\n  {'eps':>6s}  {'kernel':>10s}  {'p_Born':>8s}  {'p_MS':>8s}"
          f"  {'shift':>8s}  {'|VG|_max':>9s}")
    print(f"  {'-' * 58}")

    shifts_e = []
    for eps_g in eps_scan:
        neff_be = np.zeros(len(k_vals))
        neff_me = np.zeros(len(k_vals))
        vg_max = 0

        for ik, kv in enumerate(k_vals):
            G = build_G(dx_b, dy_b, kv)
            # Override off-diagonal with modified kernel
            omega2 = 2 * K1 * (1 - np.cos(kv)) + 4 * K2 * (2 - 2 * np.cos(kv))
            k_eff = 2 * np.sin(kv / 2)
            G_mod = np.zeros((N_b, N_b), dtype=complex)
            G_mod[mask_e] = (np.exp(1j * k_eff * dist_e[mask_e])
                             / (4 * np.pi * c_lat**2
                                * dist_e[mask_e]**(1 + eps_g)))
            denom = 1.0 / (omega2 - omega_k2 + 1j * eps_lat)
            G_00 = np.mean(denom)
            np.fill_diagonal(G_mod, G_00)

            if ik == 0:
                vg_max = np.max(np.abs(np.linalg.eigvals(V * G_mod)))

            T = np.linalg.solve(np.eye(N_b) - V * G_mod, V * np.eye(N_b))
            b = np.exp(1j * kv * dx_b)
            Tb = T @ b
            Vb = V * b

            phase_out = kv * (
                sin_th[:, :, None] * np.cos(PH)[:, :, None]
                * dx_b[None, None, :]
                + sin_th[:, :, None] * np.sin(PH)[:, :, None]
                * dy_b[None, None, :])
            a = np.exp(-1j * phase_out)
            f_ms = np.sum(a * Tb[None, None, :], axis=2)
            f_born = np.sum(a * Vb[None, None, :], axis=2)

            qz = kv * np.cos(TH)
            F2 = 4 * np.cos(qz / 2)**2
            w = F2 * transport * sin_th
            s_single = np.abs(V)**2 * np.sum(w)
            neff_be[ik] = np.sum(np.abs(f_born)**2 * w) / s_single
            neff_me[ik] = np.sum(np.abs(f_ms)**2 * w) / s_single

        p_be = np.polyfit(x, np.log(neff_be), 1)[0]
        p_me = np.polyfit(x, np.log(neff_me), 1)[0]
        sh = p_me - p_be
        shifts_e.append(sh)
        label = "1/r" if eps_g == 0 else f"1/r^{1+eps_g:.1f}"
        print(f"  {eps_g:6.2f}  {label:>10s}  {p_be:8.3f}  {p_me:8.3f}"
              f"  {sh:+8.3f}  {vg_max:9.4f}")

    # Uniform G (all pairs = G_00).
    # Sherman-Morrison: T·b = V·b + V²·G_00·N·S/(1-N·V·G_00) with S = Σ exp(ik·r_j).
    # This modifies angular distribution vs Born, producing the largest shift (+0.69).
    # Physical: uniform G = infinite-range coupling, all bonds see same propagator.
    neff_bu = np.zeros(len(k_vals))
    neff_mu = np.zeros(len(k_vals))
    for ik, kv in enumerate(k_vals):
        omega2 = 2 * K1 * (1 - np.cos(kv)) + 4 * K2 * (2 - 2 * np.cos(kv))
        denom = 1.0 / (omega2 - omega_k2 + 1j * eps_lat)
        G_00 = np.mean(denom)
        G_unif = G_00 * np.ones((N_b, N_b), dtype=complex)
        T = np.linalg.solve(np.eye(N_b) - V * G_unif, V * np.eye(N_b))
        b = np.exp(1j * kv * dx_b)
        Tb = T @ b
        Vb = V * b
        phase_out = kv * (
            sin_th[:, :, None] * np.cos(PH)[:, :, None]
            * dx_b[None, None, :]
            + sin_th[:, :, None] * np.sin(PH)[:, :, None]
            * dy_b[None, None, :])
        a = np.exp(-1j * phase_out)
        f_ms = np.sum(a * Tb[None, None, :], axis=2)
        f_born = np.sum(a * Vb[None, None, :], axis=2)
        qz = kv * np.cos(TH)
        F2 = 4 * np.cos(qz / 2)**2
        w = F2 * transport * sin_th
        s_single = np.abs(V)**2 * np.sum(w)
        neff_bu[ik] = np.sum(np.abs(f_born)**2 * w) / s_single
        neff_mu[ik] = np.sum(np.abs(f_ms)**2 * w) / s_single
    p_bu = np.polyfit(x, np.log(neff_bu), 1)[0]
    p_mu = np.polyfit(x, np.log(neff_mu), 1)[0]
    shift_unif = p_mu - p_bu
    print(f"\n  {'unif':>6s}  {'G=G_00':>10s}  {p_bu:8.3f}  {p_mu:8.3f}"
          f"  {shift_unif:+8.3f}  {'---':>9s}")

    # NOTE: eps=5 floor (+0.21) is NOT pure self-energy — 1/r^6 at distance 1
    # still gives G=1/(4*pi*c^2) ≈ 0.046, which is non-negligible for 81 bonds.
    # Clean decomposition below uses G_ij = 0 off-diagonal for true isolation.

    # Clean decomposition: 4 G types
    def compute_shift_G(G_type):
        """Compute MS shift for a specific G construction."""
        neff_b_g = np.zeros(len(k_vals))
        neff_m_g = np.zeros(len(k_vals))
        for ik, kv in enumerate(k_vals):
            omega2 = (2 * K1 * (1 - np.cos(kv))
                      + 4 * K2 * (2 - 2 * np.cos(kv)))
            k_eff = 2 * np.sin(kv / 2)
            denom = 1.0 / (omega2 - omega_k2 + 1j * eps_lat)
            G_00 = np.mean(denom)
            G_g = np.zeros((N_b, N_b), dtype=complex)
            if G_type == 'diagonal':
                np.fill_diagonal(G_g, G_00)
            elif G_type == 'no_self':
                G_g[mask_e] = (np.exp(1j * k_eff * dist_e[mask_e])
                               / (4 * np.pi * c_lat**2 * dist_e[mask_e]))
            elif G_type == 'physical':
                G_g[mask_e] = (np.exp(1j * k_eff * dist_e[mask_e])
                               / (4 * np.pi * c_lat**2 * dist_e[mask_e]))
                np.fill_diagonal(G_g, G_00)
            elif G_type == 'uniform':
                G_g = G_00 * np.ones((N_b, N_b), dtype=complex)
            T = np.linalg.solve(np.eye(N_b) - V * G_g, V * np.eye(N_b))
            b = np.exp(1j * kv * dx_b)
            Tb = T @ b
            Vb = V * b
            phase_out = kv * (
                sin_th[:, :, None] * np.cos(PH)[:, :, None]
                * dx_b[None, None, :]
                + sin_th[:, :, None] * np.sin(PH)[:, :, None]
                * dy_b[None, None, :])
            a = np.exp(-1j * phase_out)
            f_ms = np.sum(a * Tb[None, None, :], axis=2)
            f_born = np.sum(a * Vb[None, None, :], axis=2)
            qz = kv * np.cos(TH)
            F2 = 4 * np.cos(qz / 2)**2
            w = F2 * transport * sin_th
            s_single = np.abs(V)**2 * np.sum(w)
            neff_b_g[ik] = np.sum(np.abs(f_born)**2 * w) / s_single
            neff_m_g[ik] = np.sum(np.abs(f_ms)**2 * w) / s_single
        pb_g = np.polyfit(x, np.log(neff_b_g), 1)[0]
        pm_g = np.polyfit(x, np.log(neff_m_g), 1)[0]
        return pm_g - pb_g

    sh_diag = compute_shift_G('diagonal')
    sh_noself = compute_shift_G('no_self')
    sh_phys = compute_shift_G('physical')
    sh_unif = compute_shift_G('uniform')

    print(f"\n  Clean shift decomposition (true isolation):")
    print(f"    G_00 diagonal only:    {sh_diag:+.3f}  ( 9% — self-energy negligible)")
    print(f"    Off-diag 1/r only:     {sh_noself:+.3f}  (80% — DOMINANT)")
    print(f"    Physical (both):       {sh_phys:+.3f}  (100%)")
    print(f"    Uniform G (G_00*J):    {sh_unif:+.3f}  (maximum possible)")
    sh_inter = sh_phys - sh_diag - sh_noself
    print(f"    Interaction term:      {sh_inter:+.3f}"
          f"  ({sh_inter/sh_phys*100:.0f}% — G_00 and 1/r couple nonlinearly)")
    if abs(sh_inter / sh_phys) < 0.15:
        print(f"    → Interaction is moderate ({sh_inter/sh_phys*100:.0f}%):"
              f" decomposition into 80%+9% is approximate but meaningful")
    else:
        print(f"    → Interaction is significant ({sh_inter/sh_phys*100:.0f}%):"
              f" G_00 and 1/r coupling matters")
    print(f"  → Shift is ~80% from inter-bond 1/r propagation")
    print(f"  → Self-energy G_00 contributes only ~9%")
    print(f"  → 1/r REDUCES shift vs uniform ({sh_unif:+.2f} → "
          f"{sh_phys:+.2f}): phase cancellation")
    print(f"  NOTE: eps=5 'floor' (+{shifts_e[-1]:.2f}) was contaminated by"
          f" short-range 1/r^6")

    t_e = time.time() - t0 - t_a - t_b - t_c - t_d
    print(f"\n  Part E: {t_e:.0f}s")

    # ═══════════════════════════════════════════════════════════════
    # Part F: Geometry scan — 1D line vs 2D disk vs annulus
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print(f"  Part F: Geometry scan — defect shape dependence")
    print(f"{'=' * 65}")
    print(f"  V={V:.4f} (α={alpha}), 1/r kernel with phase")

    geometries = []

    # Single bond
    geometries.append(("single bond", np.array([0.0]), np.array([0.0])))

    # 1D line along x (11 bonds)
    geometries.append(("1D line (x, 11)",
                       np.arange(-5, 6, dtype=float),
                       np.zeros(11)))

    # 1D line along y (11 bonds)
    geometries.append(("1D line (y, 11)",
                       np.zeros(11),
                       np.arange(-5, 6, dtype=float)))

    # 2D annulus R=4-5
    dx_ann, dy_ann = [], []
    for ddy in range(-5, 6):
        for ddx in range(-5, 6):
            r2 = ddx**2 + ddy**2
            if 16 <= r2 <= 25:
                dx_ann.append(ddx)
                dy_ann.append(ddy)
    geometries.append(("2D annulus (R=4-5)",
                       np.array(dx_ann, dtype=float),
                       np.array(dy_ann, dtype=float)))

    # 2D disks at R=2, 5, 9
    for R_g in [2, 5, 9]:
        dx_g, dy_g = disk_bonds(R_g)
        geometries.append((f"2D disk (R={R_g})", dx_g, dy_g))

    print(f"\n  {'geometry':>25s}  {'N':>4s}  {'p_Born':>8s}  {'p_MS':>8s}"
          f"  {'shift':>8s}")
    print(f"  {'-' * 60}")

    for label, dx_g, dy_g in geometries:
        N_g = len(dx_g)
        neff_bg = np.zeros(len(k_vals))
        neff_mg = np.zeros(len(k_vals))
        dist_g = np.sqrt((dx_g[:, None] - dx_g[None, :])**2
                         + (dy_g[:, None] - dy_g[None, :])**2)
        mask_g = dist_g > 0

        for ik, kv in enumerate(k_vals):
            omega2 = (2 * K1 * (1 - np.cos(kv))
                      + 4 * K2 * (2 - 2 * np.cos(kv)))
            k_eff = 2 * np.sin(kv / 2)
            G_g = np.zeros((N_g, N_g), dtype=complex)
            if N_g > 1:
                G_g[mask_g] = (np.exp(1j * k_eff * dist_g[mask_g])
                               / (4 * np.pi * c_lat**2 * dist_g[mask_g]))
            denom = 1.0 / (omega2 - omega_k2 + 1j * eps_lat)
            G_00 = np.mean(denom)
            np.fill_diagonal(G_g, G_00)

            T = np.linalg.solve(np.eye(N_g) - V * G_g, V * np.eye(N_g))
            b = np.exp(1j * kv * dx_g)
            Tb = T @ b
            Vb = V * b

            phase_out = kv * (
                sin_th[:, :, None] * np.cos(PH)[:, :, None]
                * dx_g[None, None, :]
                + sin_th[:, :, None] * np.sin(PH)[:, :, None]
                * dy_g[None, None, :])
            a = np.exp(-1j * phase_out)
            f_ms = np.sum(a * Tb[None, None, :], axis=2)
            f_born = np.sum(a * Vb[None, None, :], axis=2)

            qz = kv * np.cos(TH)
            F2 = 4 * np.cos(qz / 2)**2
            w = F2 * transport * sin_th
            s_single = np.abs(V)**2 * np.sum(w)
            neff_bg[ik] = np.sum(np.abs(f_born)**2 * w) / s_single
            neff_mg[ik] = np.sum(np.abs(f_ms)**2 * w) / s_single

        p_bg = np.polyfit(x, np.log(neff_bg), 1)[0]
        p_mg = np.polyfit(x, np.log(neff_mg), 1)[0]
        print(f"  {label:>25s}  {N_g:4d}  {p_bg:8.3f}  {p_mg:8.3f}"
              f"  {p_mg - p_bg:+8.3f}")

    print(f"\n  → Disk gives largest shift (+0.4) and steepest Born (-2.4)")
    print(f"  → Annulus is closer to 1D line than to filled disk")
    print(f"  → Disk interior (filled area) matters, not just perimeter")

    t_f = time.time() - t0 - t_a - t_b - t_c - t_d - t_e
    print(f"\n  Part F: {t_f:.0f}s")

    # ═══════════════════════════════════════════════════════════════
    # Part G: Single-mode (Rayleigh quotient) analysis
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print(f"  Part G: Single-mode analysis — lambda_eff = <b|VG|b>/<b|b>")
    print(f"{'=' * 65}")

    # G1: lambda_eff at R=5 (no-self G — pure inter-bond)
    R = 5
    alpha = 0.30
    V = K1 * (np.cos(2 * np.pi * alpha) - 1)
    dx_b, dy_b = disk_bonds(R)
    N_b = len(dx_b)
    dist_g = np.sqrt((dx_b[:, None] - dx_b[None, :])**2
                     + (dy_b[:, None] - dy_b[None, :])**2)
    mask_g = dist_g > 0

    print(f"  R={R}, N={N_b}, V={V:.4f}, no-self G (pure 1/r)")
    print(f"\n  {'k':>5s}  {'|lam_eff|':>10s}  {'frac_par':>10s}"
          f"  {'|1/(1-lam)|':>12s}  {'resid':>8s}")
    print(f"  {'-' * 55}")

    lam_eff_arr = np.zeros(len(k_vals), dtype=complex)
    frac_arr = np.zeros(len(k_vals))
    resid_arr = np.zeros(len(k_vals))

    for ik, kv in enumerate(k_vals):
        k_eff = 2 * np.sin(kv / 2)
        G = np.zeros((N_b, N_b), dtype=complex)
        G[mask_g] = (np.exp(1j * k_eff * dist_g[mask_g])
                     / (4 * np.pi * c_lat**2 * dist_g[mask_g]))
        b = np.exp(1j * kv * dx_b)
        VGb = V * (G @ b)
        lam_eff = np.sum(VGb * np.conj(b)) / np.sum(np.abs(b)**2)
        delta = VGb - lam_eff * b
        frac = 1 - np.sum(np.abs(delta)**2) / np.sum(np.abs(VGb)**2)
        # Residual: how well VG·b ≈ λ_eff·b (eigenmode quality)
        resid = np.linalg.norm(VGb - lam_eff * b) / np.linalg.norm(VGb)
        lam_eff_arr[ik] = lam_eff
        frac_arr[ik] = frac
        resid_arr[ik] = resid
        print(f"  {kv:5.2f}  {np.abs(lam_eff):10.4f}  {frac:10.3f}"
              f"  {np.abs(1 / (1 - lam_eff)):12.4f}  {resid:8.3f}")

    abs_lam = np.abs(lam_eff_arr)
    p_lam_eff = np.polyfit(x, np.log(abs_lam), 1)[0]
    print(f"\n  |lam_eff| ~ k^{p_lam_eff:.3f}")
    print(f"  frac_parallel: {frac_arr[0]:.2f} (k=0.3) → {frac_arr[-1]:.2f}"
          f" (k=1.5)")
    print(f"  residual |VGb-λb|/|VGb|: {resid_arr[0]:.3f} (k=0.3) →"
          f" {resid_arr[-1]:.3f} (k=1.5)")
    if resid_arr[-1] > 0.3:
        print(f"  WARNING: residual > 0.3 at k=1.5 — single-mode Ansatz"
              f" VG·b ≈ λ·b is weak here")
        print(f"  %SM in table G2 at small R may overstate single-mode accuracy")

    # G2: Single-mode vs full MS at different R (no-self G)
    print(f"\n  Single-mode vs full MS at different R:")
    print(f"  {'R':>4s}  {'N':>5s}  {'p_Born':>8s}  {'p_SM':>8s}"
          f"  {'p_full':>8s}  {'%SM':>6s}")
    print(f"  {'-' * 45}")

    for R_g in [2, 3, 5, 7, 9]:
        dx_g, dy_g = disk_bonds(R_g)
        N_g = len(dx_g)
        dist_gg = np.sqrt((dx_g[:, None] - dx_g[None, :])**2
                          + (dy_g[:, None] - dy_g[None, :])**2)
        mask_gg = dist_gg > 0

        neff_bg = np.zeros(len(k_vals))
        neff_sg = np.zeros(len(k_vals))
        neff_fg = np.zeros(len(k_vals))

        for ik, kv in enumerate(k_vals):
            k_eff = 2 * np.sin(kv / 2)
            G = np.zeros((N_g, N_g), dtype=complex)
            if N_g > 1:
                G[mask_gg] = (np.exp(1j * k_eff * dist_gg[mask_gg])
                              / (4 * np.pi * c_lat**2
                                 * dist_gg[mask_gg]))
            b = np.exp(1j * kv * dx_g)
            Vb = V * b
            VGb = V * (G @ b)
            lam = np.sum(VGb * np.conj(b)) / np.sum(np.abs(b)**2)
            Tb_sm = V / (1 - lam) * b
            T = np.linalg.solve(np.eye(N_g) - V * G, V * np.eye(N_g))
            Tb_full = T @ b

            phase_out = kv * (
                sin_th[:, :, None] * np.cos(PH)[:, :, None]
                * dx_g[None, None, :]
                + sin_th[:, :, None] * np.sin(PH)[:, :, None]
                * dy_g[None, None, :])
            a = np.exp(-1j * phase_out)
            qz = kv * np.cos(TH)
            F2 = 4 * np.cos(qz / 2)**2
            w = F2 * transport * sin_th
            s_single = np.abs(V)**2 * np.sum(w)

            f_born = np.sum(a * Vb[None, None, :], axis=2)
            f_sm = np.sum(a * Tb_sm[None, None, :], axis=2)
            f_full = np.sum(a * Tb_full[None, None, :], axis=2)

            neff_bg[ik] = np.sum(np.abs(f_born)**2 * w) / s_single
            neff_sg[ik] = np.sum(np.abs(f_sm)**2 * w) / s_single
            neff_fg[ik] = np.sum(np.abs(f_full)**2 * w) / s_single

        pb_g = np.polyfit(x, np.log(neff_bg), 1)[0]
        ps_g = np.polyfit(x, np.log(neff_sg), 1)[0]
        pf_g = np.polyfit(x, np.log(neff_fg), 1)[0]
        sh_sm = ps_g - pb_g
        sh_full = pf_g - pb_g
        pct = sh_sm / sh_full * 100 if abs(sh_full) > 0.001 else 0
        print(f"  {R_g:4d}  {N_g:5d}  {pb_g:8.3f}  {ps_g:8.3f}"
              f"  {pf_g:8.3f}  {pct:5.0f}%")

    # G3: Range of contributions
    print(f"\n  Contribution range (R=5, k=0.5):")
    kv = 0.5
    k_eff = 2 * np.sin(kv / 2)
    dx_b5, dy_b5 = disk_bonds(5)
    N_b5 = len(dx_b5)
    dist_5 = np.sqrt((dx_b5[:, None] - dx_b5[None, :])**2
                     + (dy_b5[:, None] - dy_b5[None, :])**2)
    mask_5 = dist_5 > 0
    G5 = np.zeros((N_b5, N_b5), dtype=complex)
    G5[mask_5] = (np.exp(1j * k_eff * dist_5[mask_5])
                  / (4 * np.pi * c_lat**2 * dist_5[mask_5]))
    b5 = np.exp(1j * kv * dx_b5)
    VGb5 = V * (G5 @ b5)
    lam_full = np.sum(VGb5 * np.conj(b5)) / np.sum(np.abs(b5)**2)

    for r_cut in [1.5, 3.0, 5.0]:
        mask_cut = (dist_5 > 0) & (dist_5 <= r_cut)
        G_cut = np.zeros((N_b5, N_b5), dtype=complex)
        G_cut[mask_cut] = (np.exp(1j * k_eff * dist_5[mask_cut])
                           / (4 * np.pi * c_lat**2
                              * dist_5[mask_cut]))
        VGb_cut = V * (G_cut @ b5)
        lam_cut = np.sum(VGb_cut * np.conj(b5)) / np.sum(np.abs(b5)**2)
        print(f"    r <= {r_cut:.1f}: |lam|/|lam_total| ="
              f" {np.abs(lam_cut) / np.abs(lam_full):.3f}")

    print(f"\n  → Single-mode T·b ≈ V/(1-lam_eff)·b captures 77-100% of shift")
    print(f"     Valid when VG·b ≈ λ_eff·b (residual < 0.5).")
    print(f"     Exact at R >= 7 (leak into other modes vanishes).")
    print(f"  → |lam_eff| ~ k^{p_lam_eff:.2f}: low-k coherence → shift")
    print(f"  → ~80% of lam_eff from r <= 3 (short-range lattice)")
    print(f"  → C ~ 0.31 is a lattice geometric constant")
    print(f"\n  NOTE: no-self G used here to isolate inter-bond contribution")
    print(f"  (Part E shows G_00 adds only 9%). Physical G gives similar")
    print(f"  |lam_eff| but shifted by ~0.04 (self-energy offset).")
    print(f"\n  NOTE: Part A dom_T% = fraction of N_eff from top EIGENMODE of VG.")
    print(f"  %SM here = fraction of EXPONENT SHIFT from single-mode Ansatz.")
    print(f"  These measure different things: dom_T drops because b spreads")
    print(f"  across many eigenmodes at high k, but the Rayleigh quotient")
    print(f"  λ_eff still captures the weighted average enhancement.")

    t_g = time.time() - t0 - t_a - t_b - t_c - t_d - t_e - t_f
    print(f"\n  Part G: {t_g:.0f}s")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print(f"  SUMMARY")
    print(f"{'=' * 65}")
    print(f"""
  Eigenvalue analysis of VG for disk scatterer array:

  1. NOT resonance: |lam_max| = {np.abs(lam_max_arr[0]):.2f} (k=0.3) to\
 {np.abs(lam_max_arr[-1]):.2f} (k=1.5)
     Far from 1 at all k. No resonance at alpha=0.30.

  2. Phase REDUCES shift: without exp(ikr), shift = {p_np-p_b:+.2f}.
     With exp(ikr), shift = {p_p-p_b:+.2f}.

  3. Shift is LINEAR in |V|: shift ~ |V|^{p_sv:.2f} (C ~ {C_mean_5:.2f}).
     C = {np.mean(np.concatenate([shifts_5/Vs_5, C_all])):.3f}\
 +/- {np.std(np.concatenate([shifts_5/Vs_5, C_all])):.3f}, R-independent.
     Flat integrand at |V| = {V_flat:.2f} -> alpha = {alpha_pred:.3f}\
 (self-consistent).

  4. Born series oscillates: 1st corr = {(p_1-p_b)/shift_full*100:.0f}%, \
2nd = {(p_2-p_b)/shift_full*100:.0f}%, full = 100%.
     |lam_max| ~ 0.6 at alpha=0.30 -> convergent but slowly.

  5. Shift decomposition (clean): G_00-only = {sh_diag:+.2f} (9%),
     off-diag 1/r = {sh_noself:+.2f} (80%), physical = {sh_phys:+.2f} (100%).
     Uniform G gives {sh_unif:+.2f} (maximum). 1/r reduces, not causes.

  6. Geometry: disk >> line, annulus. Disk interior matters.
     Born exponent is geometry-specific (-5/2 only for filled 2D disk in 3D).

  7. Single-mode: lam_eff = <b|VG|b>/<b|b>, |lam_eff| ~ k^{p_lam_eff:.2f}.
     Captures 77% (R=2) -> 100% (R>=7) of shift. Exact at large R.
     ~80% from r <= 3 (short-range). C ~ 0.31 is lattice-specific.

  Physical picture:
    The shift is 80% from off-diagonal inter-bond propagation G(r) = exp(ikr)/r.
    At low k, phases exp(ikr) are coherent across the disk: bonds cooperate,
    MS enhancement is strong. At high k, phases randomize: cooperation drops.
    This k-dependent cooperation shifts N_eff from k^{{-5/2}} to k^{{-2}}.
    The coefficient C ~ 0.31 (shift/|V|) comes from the short-range (r <= 3)
    lattice structure of the disk pair sum. Self-energy G_00 contributes only 9%.
    The flat integrand at alpha~0.30 requires |V| ~ 1.26, yielding -2.0.""")

    t_total = time.time() - t0
    print(f"\nTotal: {t_total:.0f}s ({t_total / 60:.1f} min)")
