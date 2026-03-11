"""
Route 40 (F18): NNN selective gauging — dx=0 bonds only.

Tests the phase mechanism (F12): NNN bonds with dx=0 have phase factor
e^{ik·d} = 1 (like NN z-bonds), while dx=±1 bonds have e^{±ik}.

Prediction: gauging only dx=0 NNN bonds should restore flat integrand
(CV ≈ 7.5%), while dx=±1 only should be NOT flat.

Six cases at α=0.30, R=5, L=80:
  1. NN only (81 bonds, K1=1.0) — reference
  2. NN + NNN dx=0 only (81 + ~156 bonds, K2=0.5)
  3. NN + NNN dx=0 only with K=K1 (81 + ~156 bonds, K1=1.0)
  4. NN + NNN dx=±1 only (81 + ~156 bonds, K2=0.5)
  5. NNN dx=±1 only, no NN (~156 bonds, K2=0.5) — pure phase test
  6. NN + NNN all (81 + ~312 bonds) — reference

Results (563s):

  description                     NN  NNN K_nnn   CV%     verdict    σ(0.3)    σ(1.5)
  NN only (reference)              81    0   0.5    7.5%    FLAT     40.74      2.81
  NN + NNN(dx=0, K2=0.5)          81  156   0.5   10.7%    marginal 58.84      7.00
  NN + NNN(dx=0, K=K1=1.0)        81  156   1.0   38.6%    NOT FLAT 54.98     16.34
  NN + NNN(dx=±1, K2=0.5)         81  156   0.5    6.1%    FLAT     56.62      5.01
  NNN(dx=±1) only, no NN           0  156   0.5   33.3%    NOT FLAT 38.06      1.13
  NN + NNN(all) — full NNN         81  312   0.5   24.2%    NOT FLAT 54.25      9.98

  SURPRISE: prediction from phase mechanism (F12) is INVERTED.
  dx=0 bonds (phase=1): predicted FLAT → actually 10.7-38.6% NOT FLAT.
  dx=±1 bonds (phase=e^{±ik}): predicted NOT FLAT → actually 6.1% FLAT (with NN).

  Key findings:

  1. Phase=1 does NOT guarantee flat integrand. dx=0 NNN bonds add scattering
     that grows relative to NN (ratio 1.44 at k=0.3 to 2.49 at k=1.5).
     With K=K1 magnitude, even worse: CV=38.6%.

  2. Phase=e^{±ik} + NN is FLATTER than NN alone (6.1% < 7.5%).
     But dx=±1 alone (no NN) gives CV=33.3% — flatness comes from
     NN + dx=±1 combination, not from dx=±1 alone.

  3. Magnitude matters: dx=0 with K2=0.5 gives CV=10.7%, with K1=1.0 gives
     38.6%. Not just phase — coupling strength controls flatness.

  4. dx=±1/NN ratio peaks at k=1.1 (1.905) then decreases — consistent with
     cos(k) modulation of dx=±1 amplitude. This cos(k) partially compensates
     the NN spectral shape, producing a flatter combination.

  5. Amplitude superposition ratio σ(dx0+dx1)/σ(all) = 2.1 at k=0.3 to 1.2
     at k=1.5 — large destructive interference, especially at low k.

  CONCLUSION: The phase mechanism (F12) is qualitatively insufficient.
  Form factors of different bond groups and interference between them
  determine CV. The simple "dx=0 → phase=1 → flat" argument is wrong.
  Full NNN not-flat (CV=24.2%) is a collective effect of all bond groups.

  For paper §5: "NNN/NN difference cannot be explained by a single phase
  factor on bonds. The flat integrand results from a specific combination
  of bond geometry, coupling magnitude, and interference between bond
  groups. The NN-only gauging is the minimal construction that produces
  spectral flatness."

Run: cd ST_11/src/1_foam && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../../wip/w_21_kappa/40_nnn_selective.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../../src/1_foam'))
from elastic_3d import scalar_laplacian_3d, make_damping_3d, run_fdtd_3d
from scattering_3d import (make_wave_packet_3d, make_sphere_points,
                           compute_sphere_f2, integrate_sigma_3d,
                           estimate_n_steps_3d)
from gauge_3d import precompute_disk_bonds, precompute_nnn_disk_bonds

K1, K2 = 1.0, 0.5
L = 80
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
R_LOOP = 5
ALPHA = 0.30

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
k_vals = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])


def make_selective_vortex(alpha, R_loop, L, K1=1.0, K2=0.5,
                          nnn_filter=None, K_nnn=None, include_nn=True):
    """Vortex ring with selective NNN gauging.

    nnn_filter: which NNN groups to include.
      None or 'none' = NN only
      'dx0'  = only groups with dx=0
      'dx1'  = only groups with dx=±1
      'all'  = all 4 groups
    K_nnn: spring constant for NNN bonds (default None = use K2).
    include_nn: if False, skip NN z-bonds entirely.
    """
    cz = L // 2
    iy_disk, ix_disk = precompute_disk_bonds(L, R_loop)

    nnn_groups_all = precompute_nnn_disk_bonds(L, R_loop)
    assert len(nnn_groups_all) == 4

    # Verify group ordering by checking bond displacements
    dx0_indices = []
    dx1_indices = []
    for g_idx, (iy_lo_g, ix_lo_g, iy_hi_g, ix_hi_g) in enumerate(nnn_groups_all):
        dx_vals = np.unique(ix_hi_g - ix_lo_g)
        dy_vals = np.unique(iy_hi_g - iy_lo_g)
        assert len(dx_vals) == 1 and len(dy_vals) == 1, \
            f"Group {g_idx} has mixed displacements: dx={dx_vals}, dy={dy_vals}"
        if dx_vals[0] == 0:
            dx0_indices.append(g_idx)
        else:
            dx1_indices.append(g_idx)

    assert len(dx0_indices) == 2 and len(dx1_indices) == 2, \
        f"Expected 2 dx=0 and 2 dx≠0 groups, got {len(dx0_indices)} and {len(dx1_indices)}"

    K_nnn_eff = K_nnn if K_nnn is not None else K2

    if nnn_filter is None or nnn_filter == 'none':
        nnn_groups = []
        nnn_label = "NN only"
    elif nnn_filter == 'dx0':
        nnn_groups = [nnn_groups_all[i] for i in dx0_indices]
        nnn_label = f"{'NN + ' if include_nn else ''}NNN(dx=0)"
    elif nnn_filter == 'dx1':
        nnn_groups = [nnn_groups_all[i] for i in dx1_indices]
        nnn_label = f"{'NN + ' if include_nn else ''}NNN(dx=±1)"
    elif nnn_filter == 'all':
        nnn_groups = nnn_groups_all
        nnn_label = f"{'NN + ' if include_nn else ''}NNN(all)"
    else:
        raise ValueError(f"Unknown nnn_filter: {nnn_filter}")

    n_nn = len(iy_disk) if include_nn else 0
    n_nnn = sum(len(g[0]) for g in nnn_groups)

    phi = 2.0 * np.pi * (float(alpha) % 1.0)
    cm1 = np.cos(phi) - 1.0
    s_phi = np.sin(phi)
    if abs(s_phi) < 1e-12:
        s_phi = 0.0
    if abs(cm1) < 1e-12:
        cm1 = 0.0

    iz_lo = cz - 1
    iz_hi = cz

    def force_fn(ux, uy, uz):
        fx, fy, fz = scalar_laplacian_3d(ux, uy, uz, K1, K2)

        if cm1 == 0.0 and s_phi == 0.0:
            return fx, fy, fz

        # NN z-bonds
        if include_nn and len(iy_disk) > 0:
            ux_hi = ux[iz_hi, iy_disk, ix_disk]
            uy_hi = uy[iz_hi, iy_disk, ix_disk]
            ux_lo = ux[iz_lo, iy_disk, ix_disk]
            uy_lo = uy[iz_lo, iy_disk, ix_disk]

            fx[iz_lo, iy_disk, ix_disk] += K1 * (cm1 * ux_hi - s_phi * uy_hi)
            fy[iz_lo, iy_disk, ix_disk] += K1 * (s_phi * ux_hi + cm1 * uy_hi)

            fx[iz_hi, iy_disk, ix_disk] += K1 * (cm1 * ux_lo + s_phi * uy_lo)
            fy[iz_hi, iy_disk, ix_disk] += K1 * (-s_phi * ux_lo + cm1 * uy_lo)

        # Selected NNN bonds (using K_nnn_eff)
        for iy_lo_n, ix_lo_n, iy_hi_n, ix_hi_n in nnn_groups:
            if len(iy_lo_n) == 0:
                continue

            ux_hi_n = ux[iz_hi, iy_hi_n, ix_hi_n]
            uy_hi_n = uy[iz_hi, iy_hi_n, ix_hi_n]
            ux_lo_n = ux[iz_lo, iy_lo_n, ix_lo_n]
            uy_lo_n = uy[iz_lo, iy_lo_n, ix_lo_n]

            fx[iz_lo, iy_lo_n, ix_lo_n] += K_nnn_eff * (cm1 * ux_hi_n - s_phi * uy_hi_n)
            fy[iz_lo, iy_lo_n, ix_lo_n] += K_nnn_eff * (s_phi * ux_hi_n + cm1 * uy_hi_n)

            fx[iz_hi, iy_hi_n, ix_hi_n] += K_nnn_eff * (cm1 * ux_lo_n + s_phi * uy_lo_n)
            fy[iz_hi, iy_hi_n, ix_hi_n] += K_nnn_eff * (-s_phi * ux_lo_n + cm1 * uy_lo_n)

        return fx, fy, fz

    force_fn.n_nn = n_nn
    force_fn.n_nnn = n_nnn
    force_fn.n_total = n_nn + n_nnn
    force_fn.K_nnn = K_nnn_eff
    force_fn.label = nnn_label
    return force_fn


if __name__ == '__main__':
    t0 = time.time()

    print("Route 40 (F18): NNN selective gauging — phase mechanism test")
    print(f"  α={ALPHA}, R={R_LOOP}, L={L}")
    print(f"  k = {list(k_vals)}")
    print()

    # Verify NNN group ordering
    nnn_groups_check = precompute_nnn_disk_bonds(L, R_LOOP)
    print("NNN group verification:")
    for g_idx, (iy_lo_g, ix_lo_g, iy_hi_g, ix_hi_g) in enumerate(nnn_groups_check):
        dx_val = np.unique(ix_hi_g - ix_lo_g)[0]
        dy_val = np.unique(iy_hi_g - iy_lo_g)[0]
        phase = "e^{ik}" if dx_val != 0 else "1"
        print(f"  Group {g_idx}: dx={dx_val:+d}, dy={dy_val:+d},"
              f" n={len(iy_lo_g)}, phase={phase}")
    print()

    gamma = make_damping_3d(L, DW, DS)
    iz_s, iy_s, ix_s = make_sphere_points(L, r_m, thetas, phis)

    # References (α=0)
    def force_plain(ux, uy, uz):
        return scalar_laplacian_3d(ux, uy, uz, K1, K2)

    print("Computing references...")
    t1 = time.time()
    refs = {}
    for k0 in k_vals:
        ux0, vx0 = make_wave_packet_3d(L, k0, DW + 5, sx, K1, K2)
        ns = estimate_n_steps_3d(k0, L, DW + 5, sx, r_m, DT, 50, K1, K2)
        ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma, DT, ns,
                          rec_iz=iz_s, rec_iy=iy_s, rec_ix=ix_s, rec_n=ns)
        refs[k0] = (ref, ux0, vx0, ns)
    print(f"  Done ({time.time()-t1:.0f}s)")

    # Six cases: (filter, description, K_nnn_override, include_nn)
    cases = [
        ('none', "NN only (reference)",          None, True),
        ('dx0',  "NN + NNN(dx=0, K2=0.5)",       None, True),
        ('dx0',  "NN + NNN(dx=0, K=K1=1.0)",     K1,   True),
        ('dx1',  "NN + NNN(dx=±1, K2=0.5)",      None, True),
        ('dx1',  "NNN(dx=±1) only, no NN",        None, False),
        ('all',  "NN + NNN(all) — full NNN",      None, True),
    ]

    all_results = []

    for filt, desc, K_nnn_ovr, inc_nn in cases:
        f_test = make_selective_vortex(ALPHA, R_LOOP, L, K1, K2,
                                       nnn_filter=filt,
                                       K_nnn=K_nnn_ovr,
                                       include_nn=inc_nn)
        print(f"\n{desc}")
        print(f"  {f_test.label}: {f_test.n_nn} NN + {f_test.n_nnn} NNN"
              f" = {f_test.n_total} total bonds, K_nnn={f_test.K_nnn}")

        t1 = time.time()
        sigma_tr = np.zeros(len(k_vals))
        for j, k0 in enumerate(k_vals):
            ref, ux0, vx0, ns = refs[k0]
            d = run_fdtd_3d(f_test, ux0.copy(), vx0.copy(), gamma, DT, ns,
                            rec_iz=iz_s, rec_iy=iy_s, rec_ix=ix_s, rec_n=ns)
            f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                                   ref['ux'], ref['uy'], ref['uz'], r_m)
            _, st = integrate_sigma_3d(f2, thetas, phis)
            assert st > -1e-6, f"σ_tr={st} at k={k0}, {desc}"
            sigma_tr[j] = max(st, 0.0)
        dt = time.time() - t1

        integrand = np.sin(k_vals)**2 * sigma_tr
        if np.mean(integrand) > 1e-10:
            cv = np.std(integrand) / np.mean(integrand) * 100
        else:
            cv = 0.0

        all_results.append({
            'desc': desc, 'filter': filt, 'sigma_tr': sigma_tr.copy(),
            'integrand': integrand.copy(), 'cv': cv, 'dt': dt,
            'n_nn': f_test.n_nn, 'n_nnn': f_test.n_nnn,
            'n_total': f_test.n_total, 'K_nnn': f_test.K_nnn,
        })

        print(f"  σ_tr: " + "  ".join(f"{s:.4f}" for s in sigma_tr))
        print(f"  sin²(k)·σ_tr CV = {cv:.1f}%  ({dt:.0f}s)")

    # Summary
    print()
    print("=" * 80)
    print("Summary: NNN selective gauging")
    print("=" * 80)
    print(f"  {'description':>30s}  {'NN':>4s} {'NNN':>4s} {'K_nnn':>5s}"
          f"  {'CV%':>6s}  {'verdict':>10s}  {'σ(0.3)':>8s}  {'σ(1.5)':>8s}")
    print(f"  {'-'*80}")
    for r in all_results:
        cv = r['cv']
        verdict = "FLAT" if cv < 10 else ("marginal" if cv < 20 else "NOT FLAT")
        print(f"  {r['desc']:>30s}  {r['n_nn']:4d} {r['n_nnn']:4d} {r['K_nnn']:5.1f}"
              f"  {cv:5.1f}%  {verdict:>10s}"
              f"  {r['sigma_tr'][0]:8.4f}  {r['sigma_tr'][-1]:8.4f}")

    # Ratio analysis vs NN reference (case 0)
    print()
    s_nn = all_results[0]['sigma_tr']
    for i in range(1, len(all_results)):
        s_test = all_results[i]['sigma_tr']
        ratios = s_test / s_nn
        print(f"  {all_results[i]['desc'][:25]:>25s} / NN:"
              f"  " + "  ".join(f"{r:.3f}" for r in ratios))

    # Amplitude superposition check: NN+dx0(K2) + NN+dx1(K2) vs NN+all
    # Cases: 0=NN, 1=NN+dx0(K2), 3=NN+dx1(K2), 5=NN+all
    print()
    s_dx0_K2 = all_results[1]['sigma_tr']
    s_dx1_K2 = all_results[3]['sigma_tr']
    s_all_nnn = all_results[5]['sigma_tr']
    print("  Amplitude superposition: σ(NN+dx0) + σ(NN+dx1) vs σ(NN+all)")
    print("  NOTE: σ is NOT additive — difference measures cross-terms")
    for j, k0 in enumerate(k_vals):
        sum_parts = s_dx0_K2[j] + s_dx1_K2[j]
        ratio = sum_parts / s_all_nnn[j] if s_all_nnn[j] > 1e-10 else 0
        print(f"    k={k0:.1f}: σ_dx0={s_dx0_K2[j]:.4f}  σ_dx1={s_dx1_K2[j]:.4f}"
              f"  sum={sum_parts:.4f}  σ_all={s_all_nnn[j]:.4f}  ratio={ratio:.3f}")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
