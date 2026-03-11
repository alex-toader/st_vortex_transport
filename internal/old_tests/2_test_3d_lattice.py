"""
Phase 1: Verify 3D elastic lattice module — comprehensive tests.

Tests:
  T1: Dispersion relation matches analytic formula (5 directions)
  T2: Newton 3rd law — force sum = 0 on open boundary
  T3a: Wave propagation velocity (formula vg = C_EFF cos(k/2) along [100])
  T3b: vg cross-check via numeric finite-difference of ω(k)
  T4: PML absorbs without blow-up
  T5: Self-adjointness <u,F(v)> = <v,F(u)>
  T6: Negative semi-definiteness <u,F(u)> ≤ 0
  T7: Energy conservation without PML
  T8: Shear wave (uy-polarized) propagates correctly
  T9: Direction-dependent vg: [100] vs [110] vs [111]
  T10: Spherical isotropy scan (1000 random directions)
  T11: BZ symmetry points (Γ, X, M, R) match analytic
  T12: O(k⁶) anisotropy scaling (K1=2K2 cancels O(k⁴))

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/2_test_3d_lattice.py

--- RAW OUTPUT ---

=================================================================
Phase 1: 3D Elastic Lattice — Comprehensive Verification
=================================================================
T1: Dispersion relation (5 directions, interior slice)
  [100]: w2=0.73450463 (ana=0.73450463), err=1.89e-15, std=5.77e-15 [PASS]
  [010]: w2=0.26798107 (ana=0.26798107), err=1.55e-15, std=4.06e-15 [PASS]
  [001]: w2=1.41094688 (ana=1.41094688), err=4.44e-16, std=1.04e-14 [PASS]
  [110]: w2=0.53197247 (ana=0.53197247), err=1.55e-15, std=3.26e-14 [PASS]
  [111]: w2=0.35641755 (ana=0.35641755), err=2.34e-14, std=2.88e-13 [PASS]

T2: Force sum (Newton's 3rd — holds on any graph)
  seed=42: max|sum(f)| = 2.56e-13 [PASS]
  seed=137: max|sum(f)| = 3.48e-13 [PASS]
  seed=2024: max|sum(f)| = 1.14e-13 [PASS]

T3a: Group velocity — formula + numeric self-consistency
      [100] k=0.3: vg_num=1.712602, formula=1.712602, err=3.79e-11 [PASS]
      [100] k=0.5: vg_num=1.678206, formula=1.678206, err=3.94e-11 [PASS]
      [100] k=1.0: vg_num=1.520018, formula=1.520018, err=7.89e-11 [PASS]
            [010]: vg(dk=1e-6)=1.712602, vg(dk=1e-4)=1.712602, err=6.68e-11 [PASS]
            [001]: vg(dk=1e-6)=1.712602, vg(dk=1e-4)=1.712602, err=6.68e-11 [PASS]
            [110]: vg(dk=1e-6)=1.693419, vg(dk=1e-4)=1.693419, err=2.57e-10 [PASS]
            [111]: vg(dk=1e-6)=1.706173, vg(dk=1e-4)=1.706173, err=1.50e-10 [PASS]
            [421]: vg(dk=1e-6)=1.686941, vg(dk=1e-4)=1.686941, err=1.45e-10 [PASS]

T3b: Wave packet propagation (FDTD)
  vg (numeric) = 1.712602
  dx: expected=15.0, measured=15, err=0.001 [PASS]
  Decoupling: max|uy|=0.00e+00, max|uz|=0.00e+00 [PASS]

T4: PML — absorption, stability, no reflection
  (a) Energy ratio: 0.0030 (< 0.05) [PASS]
  (b) Finite check [PASS]
  (c) Reflection at x=20: residual/initial = 0.0302 (< 0.05) [PASS]

T5: Self-adjointness
  <u,F(v)>=7.582341e+02, <v,F(u)>=7.582341e+02, diff=6.82e-13 [PASS]

T6: Negative semi-definiteness
  seed=456: <u,F(u)> = -1.3706e+05 [PASS]
  seed=789: <u,F(u)> = -1.3692e+05 [PASS]
  seed=1234: <u,F(u)> = -1.3253e+05 [PASS]

T7: Energy conservation (no PML)
  dt=0.005: E0=1655.0218, max_drift=4.17e-08 [PASS]
  dt scaling: drift(0.1)=4.79e-04, drift(0.2)=1.91e-03, ratio=3.98 (~4.0) [PASS]

T8: Shear wave (uy-polarized)
  uy peak: dx_expected=9.8, measured=10, err=0.025 [PASS]
  Decoupling: max|ux|=0.00e+00, max|uz|=0.00e+00 [PASS]

T9: Direction-dependent group velocity
  k=0.01: vg[100]=1.732029, [110]=1.732029, [111]=1.732029, spread=1.77e-08 [PASS]
  k=1.0: vg[100]=1.520018, [110]=1.525856, [111]=1.522669, spread=0.0034 [PASS]

T10: Spherical isotropy scan (1000 directions)
  k=0.01 (continuum):
    mean(c2)=2.99997500, range=[2.999975, 2.999975]
    anisotropy (max-min)/mean = 1.63e-11 (tol 5e-04) [PASS]
  k=1.0  (lattice):
    mean(c2)=2.76045947, range=[2.758204, 2.762082]
    anisotropy (max-min)/mean = 1.40e-03 (tol 1e-02) [PASS]

T11: BZ symmetry points
  G: w2=0.000000 (analytic=0.000000), err=0.00e+00 [PASS]
  X: w2=12.000000 (analytic=12.000000), err=0.00e+00 [PASS]
  M: w2=16.000000 (analytic=16.000000), err=0.00e+00 [PASS]
  R: w2=12.000000 (analytic=12.000000), err=0.00e+00 [PASS]

T12: Anisotropy scaling (K1=2K2 kills O(k4) in w2)
  K1/6 - K2/3 = 0.00e+00 (must be 0) [PASS]
  c2 spread: k=0.15->7.03e-07, k=0.3->1.13e-05
  exponent = 4.00 (expected 4.0, generic would be 2.0) [PASS]
  contrast (K1=4K2): exponent = 2.01 (expected 2.0) [PASS]

=================================================================
RESULT: 13/13 tests passed
=================================================================
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elastic_3d import scalar_laplacian_3d, make_damping_3d, run_fdtd_3d

K1, K2 = 1.0, 0.5
C_EFF = np.sqrt(K1 + 4 * K2)  # √3


def omega2_analytic(kx, ky, kz):
    """Exact analytic ω² for 3D scalar Laplacian with NN+NNN."""
    w2 = 4 * K1 * (np.sin(kx/2)**2 + np.sin(ky/2)**2 + np.sin(kz/2)**2)
    w2 += 4 * K2 * (1 - np.cos(kx) * np.cos(ky))
    w2 += 4 * K2 * (1 - np.cos(kx) * np.cos(kz))
    w2 += 4 * K2 * (1 - np.cos(ky) * np.cos(kz))
    return w2


def vg_numeric(kx, ky, kz, dk=1e-6):
    """Numeric group velocity magnitude via finite difference of ω(k)."""
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    if k < 1e-12:
        return C_EFF
    d = np.array([kx, ky, kz]) / k
    kp = k + dk/2
    km = k - dk/2
    wp = np.sqrt(omega2_analytic(kp*d[0], kp*d[1], kp*d[2]))
    wm = np.sqrt(omega2_analytic(km*d[0], km*d[1], km*d[2]))
    return (wp - wm) / dk


# ─────────────────────────────────────────────────────────
# T1: Dispersion
# ─────────────────────────────────────────────────────────

def test_dispersion():
    """T1: Plane wave eigenvalue matches analytic dispersion.

    Why this works on open boundaries: the plane wave cos(k·r) is an
    eigenfunction of the INFINITE lattice Laplacian. Interior sites
    (3 away from edges) have all 18 neighbors present, so their force
    equals the infinite-lattice value exactly. Edge sites have missing
    neighbors — we exclude them with the interior slice.
    """
    print("T1: Dispersion relation (5 directions, interior slice)")
    L = 32
    passed = True

    for kx, ky, kz, label in [
        (0.5, 0.0, 0.0, "[100]"),
        (0.0, 0.3, 0.0, "[010]"),
        (0.0, 0.0, 0.7, "[001]"),
        (0.3, 0.3, 0.0, "[110]"),
        (0.2, 0.2, 0.2, "[111]"),
    ]:
        iz, iy, ix = np.mgrid[0:L, 0:L, 0:L].astype(float)
        u = np.cos(kx * ix + ky * iy + kz * iz)
        zeros = np.zeros((L, L, L))

        fx, _, _ = scalar_laplacian_3d(u, zeros, zeros, K1, K2)

        # Interior only: 2 away from edges is enough for NNN (diagonal reach = 1)
        # Using 3 for extra safety
        s = slice(3, L-3)
        ratio = fx[s, s, s] / u[s, s, s]
        w2_numeric = -np.mean(ratio)
        w2_std = np.std(ratio)
        w2_analytic = omega2_analytic(kx, ky, kz)

        err = abs(w2_numeric - w2_analytic)
        ok = err < 1e-10 and w2_std < 1e-10
        passed &= ok
        status = "PASS" if ok else "FAIL"
        print(f"  {label}: ω²={w2_numeric:.8f} (ana={w2_analytic:.8f}), "
              f"err={err:.2e}, std={w2_std:.2e} [{status}]")

    return passed


# ─────────────────────────────────────────────────────────
# T2: Newton's 3rd law
# ─────────────────────────────────────────────────────────

def test_force_sum():
    """T2: Total force sums to zero.

    WHY this holds on open boundaries: each bond (i,j) contributes
    K(u_j - u_i) to site i and K(u_i - u_j) to site j. Per-bond sum = 0.
    Total sum = sum of per-bond sums = 0. Missing neighbors at edges
    means fewer bonds, but each existing bond still satisfies Newton 3.
    This is a graph-theoretic identity, independent of boundary type.
    """
    print("\nT2: Force sum (Newton's 3rd — holds on any graph)")
    L = 20
    passed = True

    # Test with multiple random seeds to rule out accidental cancellation
    for seed in [42, 137, 2024]:
        np.random.seed(seed)
        ux = np.random.randn(L, L, L)
        uy = np.random.randn(L, L, L)
        uz = np.random.randn(L, L, L)

        fx, fy, fz = scalar_laplacian_3d(ux, uy, uz, K1, K2)
        s_max = max(abs(np.sum(fx)), abs(np.sum(fy)), abs(np.sum(fz)))
        ok = s_max < 1e-10
        passed &= ok
        status = "PASS" if ok else "FAIL"
        print(f"  seed={seed}: max|sum(f)| = {s_max:.2e} [{status}]")

    return passed


# ─────────────────────────────────────────────────────────
# T3a,b: Group velocity
# ─────────────────────────────────────────────────────────

def test_vg_formula():
    """T3a: Verify group velocity.

    Two sub-tests:
    (a) Along [100]: exact formula vg = C_EFF·cos(k/2).
        Derivation: ω²(k,0,0) = 4(K1+4K2)sin²(k/2) → vg = dω/dk = C_EFF·cos(k/2).
    (b) All directions: self-consistency of numeric derivative at two dk values.
        vg(dk=1e-6) ≈ vg(dk=1e-4) to high precision.
    """
    print("\nT3a: Group velocity — formula + numeric self-consistency")
    passed = True

    # (a) [100] formula: vg = C_EFF cos(k/2), exact along principal axes
    for kx, ky, kz, label in [
        (0.3, 0.0, 0.0, "[100] k=0.3"),
        (0.5, 0.0, 0.0, "[100] k=0.5"),
        (1.0, 0.0, 0.0, "[100] k=1.0"),
    ]:
        k = np.sqrt(kx**2 + ky**2 + kz**2)
        vg_num = vg_numeric(kx, ky, kz)
        vg_formula = np.cos(k / 2) * C_EFF
        err = abs(vg_num - vg_formula) / vg_num
        ok = err < 1e-6
        passed &= ok
        status = "PASS" if ok else "FAIL"
        print(f"  {label:>15s}: vg_num={vg_num:.6f}, formula={vg_formula:.6f}, "
              f"err={err:.2e} [{status}]")

    # (b) Self-consistency: numeric derivative at dk=1e-6 vs dk=1e-4
    for kx, ky, kz, label in [
        (0.0, 0.3, 0.0, "[010]"),
        (0.0, 0.0, 0.3, "[001]"),
        (0.3, 0.3, 0.0, "[110]"),
        (0.2, 0.2, 0.2, "[111]"),
        (0.4, 0.2, 0.1, "[421]"),
    ]:
        vg_fine = vg_numeric(kx, ky, kz, dk=1e-6)
        vg_coarse = vg_numeric(kx, ky, kz, dk=1e-4)
        err = abs(vg_fine - vg_coarse) / vg_fine
        ok = err < 1e-6
        passed &= ok
        status = "PASS" if ok else "FAIL"
        print(f"  {label:>15s}: vg(dk=1e-6)={vg_fine:.6f}, vg(dk=1e-4)={vg_coarse:.6f}, "
              f"err={err:.2e} [{status}]")

    return passed


def test_wave_propagation():
    """T3b: Wave packet FDTD propagation matches expected displacement."""
    print("\nT3b: Wave packet propagation (FDTD)")
    L = 60
    k0 = 0.3
    sx = 8.0
    x_start = 12.0
    dt = 0.25

    # Group velocity along [100] — use numeric for certainty
    vg = vg_numeric(k0, 0, 0)
    omega_k = np.sqrt(omega2_analytic(k0, 0, 0))

    # Make wave packet propagating in +x
    iz, iy, ix = np.mgrid[0:L, 0:L, 0:L].astype(float)
    env = np.exp(-((ix - x_start)**2) / (2 * sx**2))
    ux0 = (env * np.cos(k0 * ix)).astype(float)
    vx0 = (omega_k * env * np.sin(k0 * ix)).astype(float)

    gamma = np.zeros((L, L, L))
    n_steps = int(15.0 / (vg * dt))
    force_fn = lambda ux, uy, uz: scalar_laplacian_3d(ux, uy, uz, K1, K2)
    ux_f, uy_f, uz_f = run_fdtd_3d(force_fn, ux0, vx0, gamma, dt, n_steps)

    # Peak tracking
    cy = L // 2
    line_0 = np.abs(ux0[cy, cy, :])
    line_f = np.abs(ux_f[cy, cy, :])
    x0_peak = np.argmax(line_0)
    xf_peak = np.argmax(line_f)
    dx_measured = xf_peak - x0_peak
    dx_expected = vg * n_steps * dt

    err = abs(dx_measured - dx_expected) / dx_expected
    ok = err < 0.10  # 10% (discrete peak finding on grid)
    status = "PASS" if ok else "FAIL"
    print(f"  vg (numeric) = {vg:.6f}")
    print(f"  dx: expected={dx_expected:.1f}, measured={dx_measured}, err={err:.3f} [{status}]")

    # Component decoupling: uy, uz must stay zero
    uy_max = np.max(np.abs(uy_f))
    uz_max = np.max(np.abs(uz_f))
    ok2 = uy_max < 1e-14 and uz_max < 1e-14
    status2 = "PASS" if ok2 else "FAIL"
    print(f"  Decoupling: max|uy|={uy_max:.2e}, max|uz|={uz_max:.2e} [{status2}]")

    return ok and ok2


# ─────────────────────────────────────────────────────────
# T4: PML
# ─────────────────────────────────────────────────────────

def test_pml():
    """T4: PML absorbs wave without blow-up or reflection.

    Three sub-tests:
    (a) Energy absorption: total energy drops below 5% after 2.5 traversals.
    (b) No blow-up: all values finite.
    (c) No reflection: probe behind source after round-trip time.
        Wave launched in +x from center. Probe at x_probe (behind source).
        Any signal at probe after round-trip time must be PML reflection.

    Geometry (L=60, DW=15): PML at r > 15 from center (29.5).
    Along center line: PML at x < 14.5 and x > 44.5.
    Source at x=30 (center), probe at x=20.
    Gaussian 3σ = 12, so tail stays within [18, 42] — clear of PML.

    PML tuning: DS=1.5 chosen from parameter scan. Too weak (DS<0.5)
    → insufficient absorption. Too strong (DS>2.0) → impedance mismatch
    at PML entrance causes reflection. DW=15 gives enough ramp length
    for gradual impedance transition.
    """
    print("\nT4: PML — absorption, stability, no reflection")
    L = 60
    DW = 15
    DS = 1.5
    k0 = 0.5
    sx = 4.0
    x_start = 30.0  # center of domain
    x_probe = 20    # behind source, in PML-free zone
    dt = 0.25
    cy = L // 2

    gamma = make_damping_3d(L, DW, DS)
    assert gamma[cy, cy, cy] == 0.0, "PML nonzero at center"

    iz, iy, ix = np.mgrid[0:L, 0:L, 0:L].astype(float)
    env = np.exp(-((ix - x_start)**2) / (2 * sx**2))
    omega_k = np.sqrt(omega2_analytic(k0, 0, 0))
    ux0 = (env * np.cos(k0 * ix)).astype(float)
    vx0 = (omega_k * env * np.sin(k0 * ix)).astype(float)

    amp_init = np.max(np.abs(ux0[cy, cy, :]))
    vg = vg_numeric(k0, 0, 0)
    force_fn = lambda ux, uy, uz: scalar_laplacian_3d(ux, uy, uz, K1, K2)

    # (a) Energy absorption after 2.5 traversals from source to +x edge
    t_traverse = (L - x_start) / vg
    n_steps_a = int(2.5 * t_traverse / dt)
    ux_a, _, _ = run_fdtd_3d(force_fn, ux0, vx0, gamma, dt, n_steps_a)

    e_init = np.sum(ux0**2)
    e_final = np.sum(ux_a**2)
    ratio = e_final / e_init
    ok1 = ratio < 0.05
    status1 = "PASS" if ok1 else "FAIL"
    print(f"  (a) Energy ratio: {ratio:.4f} (< 0.05) [{status1}]")

    # (b) No blow-up
    ok2 = np.all(np.isfinite(ux_a))
    status2 = "PASS" if ok2 else "FAIL"
    print(f"  (b) Finite check [{status2}]")

    # (c) No reflection: probe at x_probe after round-trip time.
    # Extract PML edge from actual gamma array (stays in sync with make_damping_3d).
    # Along center line y=cy, z=cy: find last x where gamma=0.
    pml_profile = gamma[cy, cy, :]
    pml_edge = np.max(np.where(pml_profile == 0.0)[0])  # last non-PML site on +x
    t_roundtrip = ((pml_edge - x_start) + (pml_edge - x_probe)) / vg
    n_steps_c = int(2.0 * t_roundtrip / dt)
    ux_c, _, _ = run_fdtd_3d(force_fn, ux0, vx0, gamma, dt, n_steps_c)

    # At probe: initial wave has long passed (traveling +x).
    # Any amplitude there is reflected wave.
    residual = np.max(np.abs(ux_c[cy, cy, x_probe-2:x_probe+3]))
    reflection_ratio = residual / amp_init
    ok3 = reflection_ratio < 0.05
    status3 = "PASS" if ok3 else "FAIL"
    print(f"  (c) Reflection at x={x_probe}: residual/initial = {reflection_ratio:.4f} "
          f"(< 0.05) [{status3}]")

    return ok1 and ok2 and ok3


# ─────────────────────────────────────────────────────────
# T5, T6: Algebraic properties
# ─────────────────────────────────────────────────────────

def test_self_adjoint():
    """T5: <u, F(v)> = <v, F(u)>."""
    print("\nT5: Self-adjointness")
    L = 16
    np.random.seed(123)
    ux = np.random.randn(L, L, L)
    uy = np.random.randn(L, L, L)
    uz = np.random.randn(L, L, L)
    vx = np.random.randn(L, L, L)
    vy = np.random.randn(L, L, L)
    vz = np.random.randn(L, L, L)

    fux, fuy, fuz = scalar_laplacian_3d(ux, uy, uz, K1, K2)
    fvx, fvy, fvz = scalar_laplacian_3d(vx, vy, vz, K1, K2)

    uFv = np.sum(ux*fvx + uy*fvy + uz*fvz)
    vFu = np.sum(vx*fux + vy*fuy + vz*fuz)
    diff = abs(uFv - vFu)
    rel = diff / abs(uFv)
    ok = diff < 1e-10
    status = "PASS" if ok else "FAIL"
    print(f"  <u,F(v)>={uFv:.6e}, <v,F(u)>={vFu:.6e}, diff={diff:.2e} [{status}]")
    return ok


def test_negative_definite():
    """T6: <u, F(u)> ≤ 0."""
    print("\nT6: Negative semi-definiteness")
    L = 16
    passed = True
    for seed in [456, 789, 1234]:
        np.random.seed(seed)
        ux = np.random.randn(L, L, L)
        uy = np.random.randn(L, L, L)
        uz = np.random.randn(L, L, L)
        fx, fy, fz = scalar_laplacian_3d(ux, uy, uz, K1, K2)
        uFu = np.sum(ux*fx + uy*fy + uz*fz)
        ok = uFu <= 0
        passed &= ok
        status = "PASS" if ok else "FAIL"
        print(f"  seed={seed}: <u,F(u)> = {uFu:.4e} [{status}]")
    return passed


# ─────────────────────────────────────────────────────────
# T7: Energy conservation
# ─────────────────────────────────────────────────────────

def test_energy_conservation():
    """T7: Total energy conserved without PML.

    E = KE + PE = (1/2)Σ v² - (1/2)<u,F(u)>.
    Verlet is symplectic: energy oscillates at O(dt²) but does not drift.

    Key constraints:
    1. Wave must NOT reach boundaries (open BC leak energy).
    2. Verlet energy oscillation is O(dt²) — need small dt.

    Two sub-tests:
    (a) dt=0.005, n_steps=30: total_time=0.15, travel=0.26 sites from x=20. Safe.
    (b) dt scaling at dt=0.1 (n=30, travel=5.2) and dt=0.2 (n=15, travel=5.2).
        Both start at x=20 on L=40, max reach x≈25. Safe from edges.
    """
    print("\nT7: Energy conservation (no PML)")
    L = 40
    k0 = 0.3
    sx = 4.0
    x_start = 20.0  # center of domain
    force_fn = lambda ux, uy, uz: scalar_laplacian_3d(ux, uy, uz, K1, K2)

    def make_initial():
        iz, iy, ix = np.mgrid[0:L, 0:L, 0:L].astype(float)
        env = np.exp(-((ix - x_start)**2) / (2 * sx**2))
        omega_k = np.sqrt(omega2_analytic(k0, 0, 0))
        ux = (env * np.cos(k0 * ix)).copy()
        vx = (omega_k * env * np.sin(k0 * ix)).copy()
        return ux, vx

    def energy(ux, uy, uz, vx, vy, vz):
        ke = 0.5 * np.sum(vx**2 + vy**2 + vz**2)
        fx, fy, fz = force_fn(ux, uy, uz)
        pe = -0.5 * np.sum(ux*fx + uy*fy + uz*fz)
        return ke + pe

    def run_and_measure_drift(dt, n_steps):
        ux, vx = make_initial()
        uy = np.zeros((L, L, L)); uz = np.zeros((L, L, L))
        vy = np.zeros((L, L, L)); vz = np.zeros((L, L, L))
        e0 = energy(ux, uy, uz, vx, vy, vz)
        ax, ay, az = force_fn(ux, uy, uz)
        e_max_dev = 0.0
        for step in range(n_steps):
            ux += dt * vx + 0.5 * dt**2 * ax
            uy += dt * vy + 0.5 * dt**2 * ay
            uz += dt * vz + 0.5 * dt**2 * az
            ax_n, ay_n, az_n = force_fn(ux, uy, uz)
            vx += 0.5 * dt * (ax + ax_n)
            vy += 0.5 * dt * (ay + ay_n)
            vz += 0.5 * dt * (az + az_n)
            ax, ay, az = ax_n, ay_n, az_n
            if step % 10 == 9:
                e = energy(ux, uy, uz, vx, vy, vz)
                e_max_dev = max(e_max_dev, abs(e - e0) / e0)
        return e0, e_max_dev

    # (a) Small dt: energy conserved to high precision
    # Total time = 30 * 0.005 = 0.15. Wave travels 0.26 sites. Safe.
    # Verlet drift ∝ dt²: at dt=0.005, expect ~(0.005/0.02)² × 8e-6 ≈ 5e-7.
    e0, drift_small = run_and_measure_drift(dt=0.005, n_steps=30)
    ok1 = drift_small < 1e-6
    status1 = "PASS" if ok1 else "FAIL"
    print(f"  dt=0.005: E0={e0:.4f}, max_drift={drift_small:.2e} [{status1}]")

    # (b) dt scaling: error should go as dt²
    _, drift_01 = run_and_measure_drift(dt=0.1, n_steps=30)
    _, drift_02 = run_and_measure_drift(dt=0.2, n_steps=15)
    ratio = drift_02 / drift_01 if drift_01 > 0 else float('nan')
    # Expected ratio: (0.2/0.1)² = 4.0
    ok2 = 2.5 < ratio < 6.0
    status2 = "PASS" if ok2 else "FAIL"
    print(f"  dt scaling: drift(0.1)={drift_01:.2e}, drift(0.2)={drift_02:.2e}, "
          f"ratio={ratio:.2f} (expected ~4.0) [{status2}]")

    return ok1 and ok2


# ─────────────────────────────────────────────────────────
# T8: Shear wave
# ─────────────────────────────────────────────────────────

def test_shear_wave():
    """T8: uy-polarized wave propagates at same speed as ux-polarized.

    On the scalar Laplacian, all components are independent and identical.
    A wave in uy should propagate exactly like one in ux.
    Tests the uy0 parameter of run_fdtd_3d.
    """
    print("\nT8: Shear wave (uy-polarized)")
    L = 40
    k0 = 0.4
    sx = 5.0
    x_start = 10.0
    dt = 0.25

    omega_k = np.sqrt(omega2_analytic(k0, 0, 0))
    vg = vg_numeric(k0, 0, 0)

    iz, iy, ix = np.mgrid[0:L, 0:L, 0:L].astype(float)
    env = np.exp(-((ix - x_start)**2) / (2 * sx**2))
    uy0 = (env * np.cos(k0 * ix)).astype(float)
    vy0 = (omega_k * env * np.sin(k0 * ix)).astype(float)

    gamma = np.zeros((L, L, L))
    n_steps = int(10.0 / (vg * dt))

    force_fn = lambda ux, uy, uz: scalar_laplacian_3d(ux, uy, uz, K1, K2)

    # Initialize with ux0=0, uy0=shear wave
    ux0_zero = np.zeros((L, L, L))
    vx0_zero = np.zeros((L, L, L))
    result = run_fdtd_3d(force_fn, ux0_zero, vx0_zero, gamma, dt, n_steps,
                         uy0=uy0, vy0=vy0)
    ux_f, uy_f, uz_f = result

    # Peak tracking on uy
    cy = L // 2
    line_0 = np.abs(uy0[cy, cy, :])
    line_f = np.abs(uy_f[cy, cy, :])
    x0_peak = np.argmax(line_0)
    xf_peak = np.argmax(line_f)
    dx_measured = xf_peak - x0_peak
    dx_expected = vg * n_steps * dt

    err = abs(dx_measured - dx_expected) / max(dx_expected, 1)
    ok1 = err < 0.15
    status1 = "PASS" if ok1 else "FAIL"
    print(f"  uy peak: dx_expected={dx_expected:.1f}, measured={dx_measured}, err={err:.3f} [{status1}]")

    # ux, uz should remain zero (decoupling)
    ux_max = np.max(np.abs(ux_f))
    uz_max = np.max(np.abs(uz_f))
    ok2 = ux_max < 1e-14 and uz_max < 1e-14
    status2 = "PASS" if ok2 else "FAIL"
    print(f"  Decoupling: max|ux|={ux_max:.2e}, max|uz|={uz_max:.2e} [{status2}]")

    return ok1 and ok2


# ─────────────────────────────────────────────────────────
# T9: Direction-dependent vg
# ─────────────────────────────────────────────────────────

def test_vg_directions():
    """T9: Group velocity varies with direction at finite k (lattice anisotropy).

    At k→0: vg = C_EFF for all directions (isotropic).
    At finite k: vg depends on direction (lattice anisotropy from O(k⁴) terms).
    This test verifies that the analytic dispersion gives the expected behavior.
    """
    print("\nT9: Direction-dependent group velocity")
    passed = True

    # At small k: all directions same
    k_small = 0.01
    vg_100 = vg_numeric(k_small, 0, 0)
    vg_110 = vg_numeric(k_small/np.sqrt(2), k_small/np.sqrt(2), 0)
    vg_111 = vg_numeric(k_small/np.sqrt(3), k_small/np.sqrt(3), k_small/np.sqrt(3))

    spread_small = (max(vg_100, vg_110, vg_111) - min(vg_100, vg_110, vg_111)) / C_EFF
    ok1 = spread_small < 1e-6
    status1 = "PASS" if ok1 else "FAIL"
    print(f"  k={k_small}: vg[100]={vg_100:.6f}, [110]={vg_110:.6f}, "
          f"[111]={vg_111:.6f}, spread={spread_small:.2e} [{status1}]")
    passed &= ok1

    # At large k: anisotropy visible but bounded
    k_large = 1.0
    vg_100_L = vg_numeric(k_large, 0, 0)
    vg_110_L = vg_numeric(k_large/np.sqrt(2), k_large/np.sqrt(2), 0)
    vg_111_L = vg_numeric(k_large/np.sqrt(3), k_large/np.sqrt(3), k_large/np.sqrt(3))

    spread_large = (max(vg_100_L, vg_110_L, vg_111_L)
                    - min(vg_100_L, vg_110_L, vg_111_L)) / C_EFF
    ok2 = spread_large < 0.05  # <5% anisotropy at k=1
    status2 = "PASS" if ok2 else "FAIL"
    print(f"  k={k_large}: vg[100]={vg_100_L:.6f}, [110]={vg_110_L:.6f}, "
          f"[111]={vg_111_L:.6f}, spread={spread_large:.4f} [{status2}]")
    passed &= ok2

    return passed


# ─────────────────────────────────────────────────────────
# T10: Spherical isotropy scan
# ─────────────────────────────────────────────────────────

def test_spherical_isotropy():
    """T10: Phase velocity isotropy over 1000 random directions on sphere.

    At small k (continuum limit): c² = K1 + 4K2 for ALL directions.
    At finite k (lattice scale): O(k⁴) anisotropy from cubic symmetry.

    Generates 1000 uniformly distributed directions on the unit sphere,
    computes c²(k̂) = ω²(k·k̂)/k² for each, and reports statistics.
    """
    print("\nT10: Spherical isotropy scan (1000 directions)")
    np.random.seed(777)
    N_dirs = 1000
    c2_expected = K1 + 4 * K2
    passed = True

    # Uniform directions on sphere via Marsaglia method
    dirs = []
    while len(dirs) < N_dirs:
        v = np.random.randn(3)
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            dirs.append(v / norm)
    dirs = np.array(dirs)

    for k_mag, label, tol_spread in [
        (0.01, "k=0.01 (continuum)", 5e-4),
        (1.0,  "k=1.0  (lattice)",   0.01),
    ]:
        c2_vals = np.zeros(N_dirs)
        for i in range(N_dirs):
            kv = k_mag * dirs[i]
            w2 = omega2_analytic(kv[0], kv[1], kv[2])
            c2_vals[i] = w2 / k_mag**2

        mean = np.mean(c2_vals)
        c2_min = np.min(c2_vals)
        c2_max = np.max(c2_vals)
        spread = (c2_max - c2_min) / mean

        ok = spread < tol_spread
        passed &= ok
        status = "PASS" if ok else "FAIL"
        print(f"  {label}:")
        print(f"    mean(c²)={mean:.8f}, range=[{c2_min:.6f}, {c2_max:.6f}]")
        print(f"    anisotropy (max-min)/mean = {spread:.2e} (tol {tol_spread:.0e}) [{status}]")

    return passed


# ─────────────────────────────────────────────────────────
# T11: Brillouin zone symmetry points
# ─────────────────────────────────────────────────────────

def test_bz_symmetry_points():
    """T11: ω² at BZ symmetry points matches analytic values.

    Γ = (0,0,0):     ω² = 0
    X = (π,0,0):     ω² = 4K1 + 16K2  (NN=4K1, NNN_xy=8K2, NNN_xz=8K2, NNN_yz=0)
    M = (π,π,0):     ω² = 8K1 + 16K2  (NN=8K1, NNN_xy=0, NNN_xz=8K2, NNN_yz=8K2)
    R = (π,π,π):     ω² = 12K1         (NN=12K1, all NNN vanish: cos(π)cos(π)=1)
    """
    print("\nT11: BZ symmetry points")
    pi = np.pi
    points = [
        ("Γ", (0,0,0),       0.0),
        ("X", (pi,0,0),      4*K1 + 16*K2),
        ("M", (pi,pi,0),     8*K1 + 16*K2),
        ("R", (pi,pi,pi),    12*K1),
    ]
    passed = True
    for name, k, w2_ana in points:
        w2_num = omega2_analytic(k[0], k[1], k[2])
        err = abs(w2_num - w2_ana)
        ok = err < 1e-12
        passed &= ok
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: ω²={w2_num:.6f} (analytic={w2_ana:.6f}), err={err:.2e} [{status}]")
    return passed


# ─────────────────────────────────────────────────────────
# T12: O(k⁶) anisotropy scaling
# ─────────────────────────────────────────────────────────

def test_anisotropy_scaling():
    """T12: K1=2K2 cancels O(k⁴) anisotropy in ω², leaving O(k⁶).

    The O(k⁴) anisotropy coefficient in ω² is K1/6 - K2/3.
    With K1=1, K2=0.5: this vanishes exactly.

    Consequence for c² = ω²/k²:
      Generic (K1≠2K2): spread(c²) ∝ k⁴/k² = k²  → exponent 2
      Optimal (K1=2K2):  spread(c²) ∝ k⁶/k² = k⁴  → exponent 4
    """
    print("\nT12: Anisotropy scaling (K1=2K2 kills O(k⁴) in ω²)")

    # Verify the analytic cancellation
    coeff = K1/6 - K2/3
    ok0 = abs(coeff) < 1e-15
    status0 = "PASS" if ok0 else "FAIL"
    print(f"  K1/6 - K2/3 = {coeff:.2e} (must be 0) [{status0}]")

    # Measure c² spread exponent: should be 4 (not 2)
    np.random.seed(321)
    N_dirs = 2000
    dirs = []
    while len(dirs) < N_dirs:
        v = np.random.randn(3)
        n = np.linalg.norm(v)
        if n > 1e-10:
            dirs.append(v / n)
    dirs = np.array(dirs)

    k1, k2 = 0.15, 0.30
    spreads = []
    for k in [k1, k2]:
        c2_v = np.array([omega2_analytic(*(k*d))/k**2 for d in dirs])
        spreads.append((c2_v.max() - c2_v.min()) / c2_v.mean())

    exponent = np.log(spreads[1] / spreads[0]) / np.log(k2 / k1)
    # c² spread exponent = 4 (from O(k⁶) in ω²), generic would give 2
    ok1 = 3.5 < exponent < 4.5
    status1 = "PASS" if ok1 else "FAIL"
    print(f"  c² spread: k={k1}→{spreads[0]:.2e}, k={k2}→{spreads[1]:.2e}")
    print(f"  exponent = {exponent:.2f} (expected 4.0, generic would be 2.0) [{status1}]")

    # Contrast: verify generic K1/K2 gives exponent ~2
    K1g, K2g = 1.0, 0.25  # K1 = 4K2, not optimal
    spreads_g = []
    for k in [k1, k2]:
        c2_v = np.zeros(N_dirs)
        for i in range(N_dirs):
            kv = k * dirs[i]
            w2 = 4*K1g*(np.sin(kv[0]/2)**2 + np.sin(kv[1]/2)**2 + np.sin(kv[2]/2)**2)
            w2 += 4*K2g*(1-np.cos(kv[0])*np.cos(kv[1]))
            w2 += 4*K2g*(1-np.cos(kv[0])*np.cos(kv[2]))
            w2 += 4*K2g*(1-np.cos(kv[1])*np.cos(kv[2]))
            c2_v[i] = w2 / k**2
        spreads_g.append((c2_v.max() - c2_v.min()) / c2_v.mean())

    exp_g = np.log(spreads_g[1] / spreads_g[0]) / np.log(k2 / k1)
    ok2 = 1.5 < exp_g < 2.5
    status2 = "PASS" if ok2 else "FAIL"
    print(f"  contrast (K1=4K2): exponent = {exp_g:.2f} (expected 2.0) [{status2}]")

    return ok0 and ok1 and ok2


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Phase 1: 3D Elastic Lattice — Comprehensive Verification")
    print("=" * 65)

    results = {}
    results['T1'] = test_dispersion()
    results['T2'] = test_force_sum()
    results['T3a'] = test_vg_formula()
    results['T3b'] = test_wave_propagation()
    results['T4'] = test_pml()
    results['T5'] = test_self_adjoint()
    results['T6'] = test_negative_definite()
    results['T7'] = test_energy_conservation()
    results['T8'] = test_shear_wave()
    results['T9'] = test_vg_directions()
    results['T10'] = test_spherical_isotropy()
    results['T11'] = test_bz_symmetry_points()
    results['T12'] = test_anisotropy_scaling()

    print("\n" + "=" * 65)
    n_pass = sum(results.values())
    n_total = len(results)
    all_pass = n_pass == n_total
    print(f"RESULT: {n_pass}/{n_total} tests passed")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print("=" * 65)
