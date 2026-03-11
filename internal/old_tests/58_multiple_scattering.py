"""
Route 58: Multiple scattering mechanism — N_eff exponent from T=(I-VG)^{-1}V.

File 57 showed: Born N_eff ~ k^{-5/2} (from asymmetric forward cone), but
FDTD gives N_eff ~ k^{-2}. The correction +1/2 was empiric (C×√ω, CV=2.9%).
Single-bond T-matrix (file 57 Part G) gives wrong direction.

This file tests the collective multiple scattering hypothesis:
  - N scatterers on disk (Peierls bonds), each with V = K1·(cos2πα-1)
  - Inter-bond propagator: continuum G(r) = exp(ik_eff r)/(4πc²r)
    where k_eff = ω/c = 2sin(k/2) (continuum dispersion, NOT crystal momentum)
  - Self-energy G_00 from lattice BZ sum (retains lattice effects at r=0)
  - Full T-matrix: T = (I - V·G)^{-1} · V
  - σ_tr from angular integration of |f(θ,φ)|² with transport weight + F2

Method:
  f(k_out) = Σ_j exp(-ik_out·r_j) × (T·b)_j   (Lippmann-Schwinger)
  where b_j = exp(ik_in·r_j) is incident plane wave along x
  N_eff normalized by |V|² (Born single-bond), not |T_single|²

Parts:
  A: N_eff exponents at R=3,5,7,9 — Born vs MS vs FDTD
  B: MS/Born ratio vs √ω — does MS produce the √ω correction?
  C: Integrand flatness — CV comparison

k_eff choice verified: k_eff=ω/c=2sin(k/2) gives 100% match at R=5.
  k_eff=kv (crystal momentum) overshoots by 17%.
  Continuum dispersion is the correct choice for inter-bond propagation.

Results:

  Part A: MS exponent ≈ -2.0 at all R (Born -2.4, FDTD -2.0).
          MS shift from Born: +0.40 to +0.45. FDTD: +0.39 to +0.45.
          MS captures 91-104% of FDTD correction (V² normalization).
  Part B: CV(MS/Born/√ω) = 7-9% at all R (FDTD: 2.9%).
          C_MS has mild R-dependence (FDTD: R-independent).
  Part C: CV(sin²·σ_N_eff): Born 35.1%, MS 19.4%, FDTD 7.4%.
          MS flattens integrand partially. Remaining from lattice/vectorial.

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/58_multiple_scattering.py
"""

import numpy as np

K1, K2 = 1.0, 0.5
c_lat = np.sqrt(K1 + 4 * K2)
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
omega_arr = 2 * c_lat * np.sin(k_vals / 2)

# FDTD data (files 18, 51)
sigma_bond = np.array([0.0598, 0.0543, 0.0570, 0.0625, 0.0707, 0.0807, 0.0962])
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


# Lattice BZ grid (for self-energy G_00 only)
N_bz = 64
kx_1d = np.linspace(-np.pi, np.pi, N_bz, endpoint=False)
ky_1d = np.linspace(-np.pi, np.pi, N_bz, endpoint=False)
kz_1d = np.linspace(-np.pi, np.pi, N_bz, endpoint=False)
KX, KY, KZ = np.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
CX, CY, CZ = np.cos(KX), np.cos(KY), np.cos(KZ)
omega_k2 = (2 * K1 * (3 - CX - CY - CZ)
            + 4 * K2 * (3 - CX * CY - CY * CZ - CX * CZ))
eps_lat = 0.005

# Angular grid
N_th, N_ph = 200, 200
thetas = np.linspace(0, np.pi, N_th)
phis = np.linspace(0, 2 * np.pi, N_ph, endpoint=False)
TH, PH = np.meshgrid(thetas, phis, indexing='ij')
sin_th = np.sin(TH)
cos_ts = np.sin(TH) * np.cos(PH)
transport = 1 - cos_ts


def compute_neff_ms(dx_b, dy_b, k_vals, alpha):
    """Compute N_eff via multiple scattering T=(I-VG)^{-1}V.

    G_ij = exp(ik_eff r_ij)/(4πc²r_ij) for i≠j (continuum).
    G_ii = lattice BZ self-energy.
    k_eff = ω/c = 2sin(k/2) (continuum dispersion).
    N_eff normalized by |V|² (Born single-bond, no self-energy).
    """
    N_b = len(dx_b)
    V = K1 * (np.cos(2 * np.pi * alpha) - 1)
    dist = np.sqrt((dx_b[:, None] - dx_b[None, :])**2
                   + (dy_b[:, None] - dy_b[None, :])**2)

    neff_b = np.zeros(len(k_vals))
    neff_m = np.zeros(len(k_vals))

    for ik, kv in enumerate(k_vals):
        omega = omega_arr[ik]
        k_eff = 2 * np.sin(kv / 2)  # continuum: k = ω/c

        # G matrix: continuum off-diagonal, lattice self-energy on diagonal
        G = np.zeros((N_b, N_b), dtype=complex)
        mask = dist > 0
        G[mask] = (np.exp(1j * k_eff * dist[mask])
                   / (4 * np.pi * c_lat**2 * dist[mask]))
        denom = 1.0 / (omega**2 - omega_k2 + 1j * eps_lat)
        G_00 = np.mean(denom)
        np.fill_diagonal(G, G_00)

        # T = (I - VG)^{-1} V
        T = np.linalg.solve(np.eye(N_b) - V * G, V * np.eye(N_b))

        # Scattering amplitudes: f = Σ_j a_j (Tb)_j  (Lippmann-Schwinger)
        b = np.exp(1j * kv * dx_b)
        Tb = T @ b
        Vb = V * b

        phase_out = kv * (
            np.sin(TH)[:, :, None] * np.cos(PH)[:, :, None]
            * dx_b[None, None, :]
            + np.sin(TH)[:, :, None] * np.sin(PH)[:, :, None]
            * dy_b[None, None, :])
        a = np.exp(-1j * phase_out)

        f_ms = np.sum(a * Tb[None, None, :], axis=2)
        f_born = np.sum(a * Vb[None, None, :], axis=2)

        qz = kv * np.cos(TH)
        F2 = 4 * np.cos(qz / 2)**2
        w = F2 * transport * sin_th

        # N_eff: normalize by |V|² (Born single-bond, no self-energy)
        s_single = np.abs(V)**2 * np.sum(w)
        neff_b[ik] = np.sum(np.abs(f_born)**2 * w) / s_single
        neff_m[ik] = np.sum(np.abs(f_ms)**2 * w) / s_single

    return neff_b, neff_m


if __name__ == '__main__':
    import time
    t0 = time.time()
    alpha = 0.30
    V_pert = K1 * (np.cos(2 * np.pi * alpha) - 1)

    print("Route 58: Multiple scattering mechanism — T=(I-VG)^{-1}V")
    print(f"  α={alpha}, V={V_pert:.4f}, eps_lat={eps_lat}")
    print(f"  Angular grid: {N_th}×{N_ph}, BZ: {N_bz}³ (self-energy only)")
    print(f"  G_ij = exp(ik_eff r)/(4πc²r), k_eff = ω/c = 2sin(k/2)")
    print(f"  N_eff normalized by |V|² (Born single-bond)")
    print()

    # ═══════════════════════════════════════════════════════════════
    # Part A: N_eff exponents at R=3,5,7,9
    # ═══════════════════════════════════════════════════════════════
    print(f"{'=' * 65}")
    print(f"  Part A: N_eff exponents — Born vs MS vs FDTD")
    print(f"{'=' * 65}")

    all_neff_b = {}
    all_neff_m = {}
    all_neff_f = {}

    for R in R_vals:
        dx_b, dy_b = disk_bonds(R)
        N_b = len(dx_b)
        nb, nm = compute_neff_ms(dx_b, dy_b, k_vals, alpha)
        nf = sigma_ring_all[R] / sigma_bond

        all_neff_b[R] = nb
        all_neff_m[R] = nm
        all_neff_f[R] = nf

        p_b = np.polyfit(np.log(k_vals), np.log(nb), 1)
        p_m = np.polyfit(np.log(k_vals), np.log(nm), 1)
        p_f = np.polyfit(np.log(k_vals), np.log(nf), 1)

        print(f"\n  R={R} (N={N_b}):")
        print(f"  {'k':>5s}  {'Born':>8s}  {'MS':>8s}  {'FDTD':>8s}"
              f"  {'MS/B':>7s}  {'F/B':>7s}")
        print(f"  {'-' * 50}")
        for i in range(len(k_vals)):
            print(f"  {k_vals[i]:5.2f}  {nb[i]:8.1f}  {nm[i]:8.1f}"
                  f"  {nf[i]:8.1f}  {nm[i]/nb[i]:7.3f}"
                  f"  {nf[i]/nb[i]:7.3f}")
        print(f"  Exponents: Born={p_b[0]:.3f}  MS={p_m[0]:.3f}"
              f"  FDTD={p_f[0]:.3f}")
        print(f"  Shifts:    MS={p_m[0]-p_b[0]:+.3f}"
              f"  FDTD={p_f[0]-p_b[0]:+.3f}")

    # Summary table
    print(f"\n  {'R':>3s}  {'Born':>8s}  {'MS':>8s}  {'FDTD':>8s}"
          f"  {'MS shift':>9s}  {'FDTD shift':>10s}  {'MS/FDTD':>8s}")
    print(f"  {'-' * 58}")
    for R in R_vals:
        pb = np.polyfit(np.log(k_vals), np.log(all_neff_b[R]), 1)[0]
        pm = np.polyfit(np.log(k_vals), np.log(all_neff_m[R]), 1)[0]
        pf = np.polyfit(np.log(k_vals), np.log(all_neff_f[R]), 1)[0]
        frac = (pm - pb) / (pf - pb) if abs(pf - pb) > 0.01 else 0
        print(f"  {R:3d}  {pb:8.3f}  {pm:8.3f}  {pf:8.3f}"
              f"  {pm-pb:+9.3f}  {pf-pb:+10.3f}  {frac:8.0%}")

    # ═══════════════════════════════════════════════════════════════
    # Part B: MS/Born ratio vs √ω
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print(f"  Part B: MS/Born ratio — √ω fit")
    print(f"{'=' * 65}")

    C_ms_vals = []
    C_fdtd_vals = []
    for R in R_vals:
        ratio_m = all_neff_m[R] / all_neff_b[R]
        ratio_f = all_neff_f[R] / all_neff_b[R]
        r_sqrtw_m = ratio_m / np.sqrt(omega_arr)
        r_sqrtw_f = ratio_f / np.sqrt(omega_arr)
        C_m = np.mean(r_sqrtw_m)
        C_f = np.mean(r_sqrtw_f)
        C_ms_vals.append(C_m)
        C_fdtd_vals.append(C_f)
        print(f"\n  R={R}:")
        print(f"    CV(MS/Born):        {cv(ratio_m):5.1f}%"
              f"    CV(FDTD/Born):        {cv(ratio_f):5.1f}%")
        print(f"    CV(MS/Born/√ω):     {cv(r_sqrtw_m):5.1f}%"
              f"    CV(FDTD/Born/√ω):     {cv(r_sqrtw_f):5.1f}%")
        print(f"    C_MS = {C_m:.4f}"
              f"              C_FDTD = {C_f:.4f}")

    C_ms_arr = np.array(C_ms_vals)
    C_fdtd_arr = np.array(C_fdtd_vals)
    R_arr = np.array(R_vals, dtype=float)
    p_cm = np.polyfit(np.log(R_arr), np.log(C_ms_arr), 1)
    p_cf = np.polyfit(np.log(R_arr), np.log(C_fdtd_arr), 1)
    print(f"\n  C_MS ~ R^{{{p_cm[0]:.2f}}}   C_FDTD ~ R^{{{p_cf[0]:.2f}}}")

    # ═══════════════════════════════════════════════════════════════
    # Part C: Integrand flatness (using N_eff × σ_bond for fair comparison)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print(f"  Part C: Integrand flatness — CV(sin²k · σ_bond · N_eff)")
    print(f"{'=' * 65}")

    # Fair comparison: multiply N_eff by common σ_bond(k) from FDTD
    nb5 = all_neff_b[5]
    nm5 = all_neff_m[5]
    nf5 = all_neff_f[5]

    int_born = np.sin(k_vals)**2 * sigma_bond * nb5
    int_ms = np.sin(k_vals)**2 * sigma_bond * nm5
    int_fdtd = np.sin(k_vals)**2 * sigma_ring_all[5]

    int_born_n = int_born / int_born[0]
    int_ms_n = int_ms / int_ms[0]
    int_fdtd_n = int_fdtd / int_fdtd[0]

    print(f"\n  sin²(k) · σ_bond · N_eff (R=5, normalized to k=0.3):")
    print(f"  {'k':>5s}  {'Born':>8s}  {'MS':>8s}  {'FDTD':>8s}")
    print(f"  {'-' * 35}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {int_born_n[i]:8.4f}"
              f"  {int_ms_n[i]:8.4f}  {int_fdtd_n[i]:8.4f}")

    print(f"\n  CV: Born={cv(int_born_n):.1f}%"
          f"  MS={cv(int_ms_n):.1f}%  FDTD={cv(int_fdtd_n):.1f}%")
    print(f"  (σ_bond from FDTD, common to all three — isolates N_eff effect)")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print(f"  SUMMARY")
    print(f"{'=' * 65}")
    print(f"""
  Multiple scattering T=(I-VG)^{{-1}}V with continuum G(r):

  N_eff exponent (V² normalization):
    Born:  -5/2 ≈ -2.4   (asymmetric forward cone, file 57)
    MS:    ≈ -2.0         (shift +0.40 to +0.45 at all R)
    FDTD:  ≈ -2.0         (shift +0.39 to +0.45 at all R)
    MS/FDTD: 91-104%

  Model ingredients:
    - continuum propagator exp(i k_eff r)/(4πc²r), k_eff = ω/c = 2sin(k/2)
    - lattice self-energy G_00 (single-bond renormalization)
    - NO lattice dispersion in inter-bond G
    - NO vectorial effects (scalar model)

  CONFIRMED: the +1/2 exponent correction arises from collective
  multiple scattering between the bonds of the topological defect.
  The continuum propagator between bonds is sufficient.

  OPEN: why exactly +1/2. Requires eigenvalue analysis of VG(k)
  for disk scatterer array with G(r) = exp(ikr)/(4πr).""")

    dt = time.time() - t0
    print(f"\nTotal: {dt:.0f}s ({dt / 60:.1f} min)")
