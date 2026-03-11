"""
Route 56: N_eff structure — coherent-incoherent crossover.

N_eff = sigma_ring / sigma_bond measures how many bonds contribute
effectively to the ring cross section. This file investigates its
k and R dependence.

Key finding: N_eff ∝ 1/k² at fixed R (CV=2.3%), NOT 1/sin²(k/2) (CV=5.4%).
Born coherent sum |S(Q)|² predicts (kR)^{-2.4}, FDTD gives (kR)^{-2.0}.
FDTD exponent closer to -2 than Born's -2.4. Origin of difference open.

Model: N_eff = N² / (1 + c · k² · R^q)
  c ≈ 2.1, q ≈ 2.37
  Crossover at kR ~ 0.5: coherent (N²) → incoherent (1/k²)
  q = 2β - p_R where β≈1.96 (N∝R^β), p_R=1.6 (σ_ring∝R^{p_R}).

Connection to flat integrand:
  sin²(k)·σ_ring = 4sin²(k/2)/k² · C₀·V·N² / (c·R^q)
  sin²(k/2)/k² ≈ 1/4 (CV=6.1%), V ≈ const at α≥0.25 (CV=2.7%)
  → integrand CV ≈ 7.4%  ✓

Verification (zero-compute, uses FDTD data from files 18, 51):
  Part A: N_eff(k) at R=5 — 1/k² vs 1/sin²(k/2) comparison
  Part B: Born |S(Q)|² prediction vs FDTD N_eff
  Part C: Per-R fit of c — c depends weakly on R
  Part D: Global fit N_eff = N²/(1 + c·k²·R^q), connection q = 2β - p_R
  Part E: Full flat integrand chain with 1/k² scaling

Results (0s, zero-compute):

  Part A: k²·N_eff CV=2.3% (R=5). sin²(k/2)·N_eff CV=5.4%. 1/k² wins.
          Power law exponent = -1.97 ≈ -2.
          NOTE: kR_min=1.5 >> crossover kR≈0.5. Coherent plateau (N²) NOT sampled.
  Part B: Born N_eff has (kR)^{-2.41}. FDTD has (kR)^{-1.97}. Born/FDTD = 0.38-0.80.
          FDTD exponent closer to -2 than Born's -2.4. Origin of difference open.
  Part C: c(R): R=3→3.0, R=5→3.9, R=7→4.2, R=9→4.5. c ∝ R^{0.37}.
  Part D: Global c=2.1, q=2.37. q = 2β-p_R = 2×1.96-1.6 = 2.32 (fit: 2.37, Δ=0.05).
          N ∝ R^{1.96}. Part C c differs from Part D c (absorbs R^{q-2}).
  Part E: sin²(k/2)/k² CV=6.1%. V CV=2.7%. Integrand CV=7.4%.
          Chain [sin²(k/2)/k² × V] CV=8.7% > integrand 7.4%: partial cancellation
          between chain drift and N_eff non-Born residual. Residual CV=3.2%.

  N_eff ∝ 1/k² is EMPIRIC (non-Born). Born gives (kR)^{-2.4}, FDTD
  gives cleaner (kR)^{-2.0}. Origin of difference open.

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/56_neff_structure.py
"""

import numpy as np
from scipy.special import j1
from scipy.optimize import minimize_scalar, minimize

K1, K2 = 1.0, 0.5
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])

# FDTD per-bond data (file 51, α=0.30, single z-bond)
sigma_bond = np.array([0.0598, 0.0543, 0.0570, 0.0625, 0.0707, 0.0807, 0.0962])

# FDTD ring data from file 18 (α=0.30, R=3,5,7,9)
sigma_ring_data = {
    3: np.array([13.16, 6.43, 3.54, 2.28, 1.72, 1.43, 1.36]),
    5: np.array([40.74, 14.16, 7.69, 5.05, 3.78, 3.14, 2.81]),
    7: np.array([72.70, 22.23, 12.22, 8.16, 6.26, 5.25, 4.71]),
    9: np.array([117.84, 34.54, 19.43, 13.34, 10.29, 8.56, 7.74]),
}
R_vals = [3, 5, 7, 9]


def cv(x):
    return np.std(x) / np.abs(np.mean(x)) * 100


def count_bonds(R):
    n = 0
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            if dx * dx + dy * dy <= R * R:
                n += 1
    return n


def disk_bonds(R):
    dx, dy = [], []
    for ddy in range(-R, R + 1):
        for ddx in range(-R, R + 1):
            if ddx * ddx + ddy * ddy <= R * R:
                dx.append(ddx)
                dy.append(ddy)
    return np.array(dx, dtype=float), np.array(dy, dtype=float)


if __name__ == '__main__':
    import time
    t0 = time.time()
    print("Route 56: N_eff structure — coherent-incoherent crossover")
    print()

    N_bonds = {R: count_bonds(R) for R in R_vals}
    N_eff = {R: sigma_ring_data[R] / sigma_bond for R in R_vals}

    # ═══════════════════════════════════════════════════════════════
    # Part A: N_eff(k) at R=5 — which scaling?
    # ═══════════════════════════════════════════════════════════════
    print(f"{'=' * 60}")
    print(f"  Part A: N_eff(k) at R=5 — 1/k² vs 1/sin²(k/2)")
    print(f"{'=' * 60}")

    neff5 = N_eff[5]
    prod_k2 = k_vals**2 * neff5
    prod_sin2 = np.sin(k_vals / 2)**2 * neff5

    print(f"\n  N_bonds = {N_bonds[5]}, N² = {N_bonds[5]**2}")
    print(f"  NOTE: N_eff(k=0.3) = {neff5[0]:.0f} << N² = {N_bonds[5]**2}.")
    print(f"  Min kR = {k_vals[0]*5:.1f} >> crossover kR ≈ 0.5.")
    print(f"  Coherent plateau (N²) is NOT sampled — model valid in power-law regime only.")
    print(f"\n  {'k':>5s}  {'N_eff':>8s}  {'k²·Neff':>10s}  {'sin²·Neff':>10s}")
    print(f"  {'-' * 40}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {neff5[i]:8.1f}  {prod_k2[i]:10.2f}"
              f"  {prod_sin2[i]:10.2f}")

    print(f"\n  k²·N_eff: CV = {cv(prod_k2):.1f}%")
    print(f"  sin²(k/2)·N_eff: CV = {cv(prod_sin2):.1f}%")
    print(f"  → N_eff ∝ 1/k² (CV={cv(prod_k2):.1f}%) wins over"
          f" 1/sin²(k/2) (CV={cv(prod_sin2):.1f}%)")

    # Power law fit
    p_fit = np.polyfit(np.log(k_vals), np.log(neff5), 1)
    print(f"\n  Power law: N_eff = {np.exp(p_fit[1]):.1f} · k^{{{p_fit[0]:.3f}}}")
    print(f"  Exponent {p_fit[0]:.3f} ≈ -2")

    # ═══════════════════════════════════════════════════════════════
    # Part B: Born |S(Q)|² prediction vs FDTD
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Part B: Born |S(Q)|² vs FDTD N_eff (R=5)")
    print(f"{'=' * 60}")

    R = 5
    dx_b, dy_b = disk_bonds(R)
    N_b = len(dx_b)

    N_th, N_ph = 200, 200
    thetas = np.linspace(0, np.pi, N_th)
    phis = np.linspace(0, 2 * np.pi, N_ph, endpoint=False)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    d_th = thetas[1] - thetas[0]
    d_ph = phis[1] - phis[0]
    sin_th = np.sin(TH)
    cos_theta_s = np.sin(TH) * np.cos(PH)
    transport = 1 - cos_theta_s

    N_eff_born = np.zeros(len(k_vals))
    for ik, k in enumerate(k_vals):
        Q_x = k * np.sin(TH) * np.cos(PH) - k
        Q_y = k * np.sin(TH) * np.sin(PH)

        phase = (Q_x[np.newaxis] * dx_b[:, np.newaxis, np.newaxis]
                 + Q_y[np.newaxis] * dy_b[:, np.newaxis, np.newaxis])
        S = np.sum(np.exp(-1j * phase), axis=0)
        S2 = np.abs(S)**2

        q_z = k * np.cos(TH)
        F2 = 4 * np.cos(q_z / 2)**2
        w = F2 * transport * sin_th

        N_eff_born[ik] = (np.sum(S2 * w) * d_th * d_ph
                          / (np.sum(w) * d_th * d_ph))

    p_born = np.polyfit(np.log(k_vals * R), np.log(N_eff_born), 1)
    p_fdtd = np.polyfit(np.log(k_vals * R), np.log(neff5), 1)

    print(f"\n  {'k':>5s}  {'N_eff_Born':>10s}  {'N_eff_FDTD':>10s}  {'FDTD/Born':>10s}")
    print(f"  {'-' * 40}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {N_eff_born[i]:10.1f}  {neff5[i]:10.1f}"
              f"  {neff5[i] / N_eff_born[i]:10.3f}")

    print(f"\n  Born:  N_eff ∝ (kR)^{{{p_born[0]:.2f}}}")
    print(f"  FDTD:  N_eff ∝ (kR)^{{{p_fdtd[0]:.2f}}}")
    print(f"  Born overestimates at low k, converges at high k.")
    print(f"  FDTD exponent closer to -2 than Born's -2.4. Origin of difference open.")

    # ═══════════════════════════════════════════════════════════════
    # Part C: Per-R fit of c
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Part C: N_eff = N²/(1 + c·(kR)²), per-R fit")
    print(f"{'=' * 60}")

    c_per_R = {}
    print(f"\n  {'R':>3s}  {'N':>5s}  {'c':>8s}  {'CV%':>6s}  {'kR_cross':>8s}")
    print(f"  {'-' * 35}")
    for Rv in R_vals:
        N = N_bonds[Rv]
        kR = k_vals * Rv

        def resid(c, _N=N, _kR=kR, _neff=N_eff[Rv]):
            model = _N**2 / (1 + c * _kR**2)
            return np.sum((np.log(model) - np.log(_neff))**2)

        res = minimize_scalar(resid, bounds=(0.1, 20), method='bounded')
        c_R = res.x
        c_per_R[Rv] = c_R
        model = N**2 / (1 + c_R * kR**2)
        ratio = N_eff[Rv] / model
        print(f"  {Rv:3d}  {N:5d}  {c_R:8.3f}  {cv(ratio):5.1f}%"
              f"  {1 / np.sqrt(c_R):8.3f}")

    c_arr = np.array([c_per_R[R] for R in R_vals])
    R_arr = np.array(R_vals, dtype=float)
    p_c = np.polyfit(np.log(R_arr), np.log(c_arr), 1)
    print(f"\n  c(R) ∝ R^{{{p_c[0]:.2f}}}")
    print(f"  c increases slowly with R: not universal constant.")
    print(f"  NOTE: Part C fixes q=2 in (kR)². Part D frees q.")
    print(f"  The c values differ: Part C absorbs R^(q-2) into c.")

    # ═══════════════════════════════════════════════════════════════
    # Part D: Global fit with R^q
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Part D: Global N_eff = N²/(1 + c·k²·R^q)")
    print(f"{'=' * 60}")

    def global_resid(params):
        c, q = params
        total = 0
        for Rv in R_vals:
            N = N_bonds[Rv]
            model = N**2 / (1 + c * k_vals**2 * Rv**q)
            total += np.sum((np.log(model) - np.log(N_eff[Rv]))**2)
        return total

    res_g = minimize(global_resid, [4.0, 2.4], method='Nelder-Mead')
    c_g, q_g = res_g.x

    # N ∝ R^β — compute β first for formula
    N_arr_v = np.array([N_bonds[Rv] for Rv in R_vals], dtype=float)
    p_N = np.polyfit(np.log(R_arr), np.log(N_arr_v), 1)
    beta = p_N[0]

    print(f"\n  c = {c_g:.3f}, q = {q_g:.3f}")
    print(f"  N ∝ R^{{{beta:.2f}}} → q = 2β - p_R = "
          f"2×{beta:.2f} - 1.6 = {2*beta-1.6:.2f} (fit: {q_g:.2f})")
    print(f"  Consistency: q + p_R = {q_g:.2f} + 1.6 = {q_g+1.6:.2f}"
          f" vs 2β = {2*beta:.2f}. Δ={abs(q_g+1.6-2*beta):.2f}.")

    print(f"\n  {'R':>3s}  {'k':>5s}  {'N_eff':>8s}  {'model':>8s}  {'ratio':>8s}")
    print(f"  {'-' * 40}")
    for Rv in R_vals:
        N = N_bonds[Rv]
        model = N**2 / (1 + c_g * k_vals**2 * Rv**q_g)
        ratio = N_eff[Rv] / model
        for i in [0, 3, 6]:
            print(f"  {Rv:3d}  {k_vals[i]:5.2f}  {N_eff[Rv][i]:8.1f}"
                  f"  {model[i]:8.1f}  {ratio[i]:8.4f}")
        print(f"  R={Rv}: CV = {cv(ratio):.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # Part E: Full flat integrand chain
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Part E: Flat integrand chain with 1/k² scaling")
    print(f"{'=' * 60}")

    alpha = 0.30
    cm1 = np.cos(2 * np.pi * alpha) - 1
    s_phi = np.sin(2 * np.pi * alpha)
    v_g = np.sqrt(K1 + 4 * K2) * np.cos(k_vals / 2)
    Z_mono = 8 * np.pi * (1 + np.sin(k_vals) / k_vals)
    Z_dipo = 8 * np.pi * (1 - np.sin(k_vals) / k_vals)
    V = cm1**2 * Z_mono + s_phi**2 * Z_dipo

    integrand = np.sin(k_vals)**2 * sigma_ring_data[5]
    factor = np.sin(k_vals / 2)**2 / k_vals**2

    print(f"\n  sin²(k)·σ_ring = [sin²(k/2)/k²] × [4·C₀·V·N²/(c·R^q)]")
    print(f"\n  k-dependent part: sin²(k/2)/k²")
    print(f"  {'k':>5s}  {'sin²(k/2)/k²':>14s}  {'V':>8s}  {'integrand_n':>12s}")
    print(f"  {'-' * 45}")
    integ_n = integrand / integrand[0]
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {factor[i]:14.4f}  {V[i]:8.2f}"
              f"  {integ_n[i]:12.4f}")

    print(f"\n  sin²(k/2)/k²: CV = {cv(factor):.1f}%"
          f" (→ 1/4 = 0.250 at k→0)")
    print(f"  V (Born vertex): CV = {cv(V):.1f}%")
    print(f"  sin²(k)·σ_ring: CV = {cv(integrand):.1f}%")

    # Chain accuracy
    # sin²(k)·σ_ring = 4sin²(k/2)cos²(k/2) · σ_bond · N_eff
    # with σ_bond ∝ V/cos²(k/2) and N_eff ∝ 1/k²:
    # = 4sin²(k/2)·V·N_eff ∝ sin²(k/2)/k² · V
    chain = factor * V
    chain_n = chain / chain[0]
    resid_cv = cv(integ_n / chain_n)
    print(f"\n  Chain [sin²(k/2)/k² × V] normalized: CV = {cv(chain_n):.1f}%")
    print(f"  Residual (integrand / chain): CV = {resid_cv:.1f}%")
    print(f"  Chain CV ({cv(chain_n):.1f}%) > integrand CV ({cv(integrand):.1f}%):")
    print(f"  partial cancellation between chain drift and N_eff non-Born residual.")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"""
  N_eff = N²/(1 + c·k²·R^q)  with c≈{c_g:.1f}, q≈{q_g:.1f}

  At fixed R: N_eff ∝ 1/k² (CV={cv(prod_k2):.1f}%, better than 1/sin²(k/2) at {cv(prod_sin2):.1f}%)
  Power law exponent: FDTD gives (kR)^{{{p_fdtd[0]:.2f}}}, Born gives (kR)^{{{p_born[0]:.2f}}}
  FDTD exponent closer to -2 than Born's -2.4. Origin of difference open.

  q = 2β - p_R = 2×{beta:.2f} - 1.6 = {2*beta-1.6:.2f} (fit: {q_g:.2f})
  (β from N∝R^β, p_R=1.6 from σ_ring∝R^p_R, file 18)

  Flat integrand: sin²(k)·σ_ring ≈ const (CV={cv(integrand):.1f}%) because:
    1. cos²(k/2) cancels exactly (algebraic)
    2. N_eff ∝ 1/k² → sin²(k/2)/k² ≈ 1/4 (CV={cv(factor):.1f}%)
    3. V ≈ const at α≥0.25 (CV={cv(V):.1f}%)

  STATUS:
    DERIVED: per-bond Born, cos²(k/2) cancellation, V ≈ const
    EMPIRIC: N_eff ∝ 1/k² (non-Born, cleaner than Born)
    OPEN: why non-Born simplifies exponent from -2.4 to -2.0""")

    dt = time.time() - t0
    print(f"\nTotal: {dt:.0f}s ({dt / 60:.1f} min)")
