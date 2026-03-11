"""
Phase 2 tests for gauge_3d.py — Peierls vortex ring on 3D lattice.

6 tests:
  T1  alpha=0 identity: vortex force == scalar Laplacian
  T2  Disk bond count: matches lattice disk area
  T3  Correction locality: delta_f nonzero only at disk-adjacent layers
  T4  R/R^T structure: single-bond correction matches analytic
  T5  Holonomy: threading loop gives R(2*pi*alpha), non-threading gives I
  T6  Self-adjoint at alpha!=0: <u, F(v)> == <v, F(u)> for generic alpha
  T7  Gauge invariance: move Dirac disk, verify energy invariance

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/4_test_gauge_3d.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from elastic_3d import scalar_laplacian_3d
from gauge_3d import make_vortex_force, precompute_disk_bonds

L = 30
K1, K2 = 1.0, 0.5
R_LOOP = 5.0
CX = CY = CZ = L // 2

passed = 0
failed = 0


def report(name, ok, detail=""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  {tag}: {name}" + (f"  ({detail})" if detail else ""))


# ── T1: alpha=0 identity ──────────────────────────────────────

print("T1: alpha=0 identity")
rng = np.random.RandomState(42)
ux = rng.randn(L, L, L)
uy = rng.randn(L, L, L)
uz = rng.randn(L, L, L)

f_plain = scalar_laplacian_3d(ux, uy, uz, K1, K2)
force_fn_0 = make_vortex_force(0.0, R_LOOP, L, K1, K2)
f_vortex = force_fn_0(ux.copy(), uy.copy(), uz.copy())

for comp, name in zip(range(3), ['fx', 'fy', 'fz']):
    diff = np.max(np.abs(f_vortex[comp] - f_plain[comp]))
    report(f"{name} matches", diff < 1e-14, f"max diff = {diff:.2e}")


# ── T2: Disk bond count ───────────────────────────────────────

print("T2: Disk bond count")
iy_disk, ix_disk = precompute_disk_bonds(L, R_LOOP)
n_bonds = len(iy_disk)

# Expected: lattice points inside circle of radius R_LOOP centered at (CX, CY)
# Exact count by brute force
expected = 0
for iy in range(L):
    for ix in range(L):
        if (ix - CX)**2 + (iy - CY)**2 <= R_LOOP**2:
            expected += 1

report("bond count", n_bonds == expected, f"got {n_bonds}, expected {expected}")

# Sanity: close to pi*R^2
area_approx = np.pi * R_LOOP**2
report("close to pi*R^2", abs(n_bonds - area_approx) < 2 * R_LOOP + 5,
       f"{n_bonds} vs pi*R^2={area_approx:.1f}")


# ── T3: Correction locality ──────────────────────────────────

print("T3: Correction locality")
alpha_test = 0.3
force_fn = make_vortex_force(alpha_test, R_LOOP, L, K1, K2)

ux2 = rng.randn(L, L, L)
uy2 = rng.randn(L, L, L)
uz2 = rng.randn(L, L, L)

f_plain2 = scalar_laplacian_3d(ux2, uy2, uz2, K1, K2)
f_vortex2 = force_fn(ux2.copy(), uy2.copy(), uz2.copy())

dfx = f_vortex2[0] - f_plain2[0]
dfy = f_vortex2[1] - f_plain2[1]
dfz = f_vortex2[2] - f_plain2[2]

# fz should be zero everywhere (Option B: uz untouched)
report("dfz == 0 everywhere", np.max(np.abs(dfz)) < 1e-14,
       f"max |dfz| = {np.max(np.abs(dfz)):.2e}")

# dfx, dfy nonzero only at iz_lo and iz_hi
iz_lo = CZ - 1
iz_hi = CZ
for name, df in [('dfx', dfx), ('dfy', dfy)]:
    # Zero outside disk layers
    mask_other = np.ones(L, dtype=bool)
    mask_other[iz_lo] = False
    mask_other[iz_hi] = False
    max_other = np.max(np.abs(df[mask_other]))
    report(f"{name} zero outside disk layers", max_other < 1e-14,
           f"max = {max_other:.2e}")

    # Nonzero at disk layers (at disk bond positions)
    max_at_disk = max(np.max(np.abs(df[iz_lo, iy_disk, ix_disk])),
                      np.max(np.abs(df[iz_hi, iy_disk, ix_disk])))
    report(f"{name} nonzero at disk bonds", max_at_disk > 1e-6,
           f"max = {max_at_disk:.4f}")

    # Zero at disk layers but outside disk radius
    mask_outside = np.ones((L, L), dtype=bool)
    mask_outside[iy_disk, ix_disk] = False
    iy_out, ix_out = np.where(mask_outside)
    if len(iy_out) > 0:
        max_out_lo = np.max(np.abs(df[iz_lo, iy_out, ix_out]))
        max_out_hi = np.max(np.abs(df[iz_hi, iy_out, ix_out]))
        max_out = max(max_out_lo, max_out_hi)
        report(f"{name} zero outside disk radius", max_out < 1e-14,
               f"max = {max_out:.2e}")


# ── T4: R/R^T structure on single bond ───────────────────────

print("T4: R/R^T structure check")
# Pick one bond inside the disk: (CZ-1, CY, CX) <-> (CZ, CY, CX)
# Use a simple displacement to check R and R^T
alpha4 = 0.25
phi4 = 2 * np.pi * alpha4
c4 = np.cos(phi4) - 1.0
s4 = np.sin(phi4)

# Create field nonzero only at the two disk sites
ux4 = np.zeros((L, L, L))
uy4 = np.zeros((L, L, L))
uz4 = np.zeros((L, L, L))
ux4[iz_lo, CY, CX] = 1.0  # u_lo = (1, 0)
uy4[iz_lo, CY, CX] = 0.0
ux4[iz_hi, CY, CX] = 0.0  # u_hi = (0, 1)
uy4[iz_hi, CY, CX] = 1.0

force_fn4 = make_vortex_force(alpha4, R_LOOP, L, K1, K2)
f4 = force_fn4(ux4.copy(), uy4.copy(), uz4.copy())
f4_plain = scalar_laplacian_3d(ux4, uy4, uz4, K1, K2)

# Correction at lo from hi: K1*(R-I)*u_hi = K1*((c-1)*0 - s*1, s*0 + (c-1)*1)
#                          = K1*(-s, c-1)
expected_dfx_lo = K1 * (-s4)       # cm1*0 - s*1
expected_dfy_lo = K1 * (c4)        # s*0 + cm1*1

# Correction at hi from lo: K1*(R^T-I)*u_lo = K1*((c-1)*1 + s*0, -s*1 + (c-1)*0)
#                          = K1*(c-1, -s)
expected_dfx_hi = K1 * (c4)        # cm1*1 + s*0
expected_dfy_hi = K1 * (-s4)       # -s*1 + cm1*0

actual_dfx_lo = f4[0][iz_lo, CY, CX] - f4_plain[0][iz_lo, CY, CX]
actual_dfy_lo = f4[1][iz_lo, CY, CX] - f4_plain[1][iz_lo, CY, CX]
actual_dfx_hi = f4[0][iz_hi, CY, CX] - f4_plain[0][iz_hi, CY, CX]
actual_dfy_hi = f4[1][iz_hi, CY, CX] - f4_plain[1][iz_hi, CY, CX]

report("dfx_lo = K1*(-sin)", abs(actual_dfx_lo - expected_dfx_lo) < 1e-14,
       f"got {actual_dfx_lo:.6f}, expected {expected_dfx_lo:.6f}")
report("dfy_lo = K1*(cos-1)", abs(actual_dfy_lo - expected_dfy_lo) < 1e-14,
       f"got {actual_dfy_lo:.6f}, expected {expected_dfy_lo:.6f}")
report("dfx_hi = K1*(cos-1)", abs(actual_dfx_hi - expected_dfx_hi) < 1e-14,
       f"got {actual_dfx_hi:.6f}, expected {expected_dfx_hi:.6f}")
report("dfy_hi = K1*(-sin)", abs(actual_dfy_hi - expected_dfy_hi) < 1e-14,
       f"got {actual_dfy_hi:.6f}, expected {expected_dfy_hi:.6f}")


# ── T5: Holonomy via force-extracted gauge ────────────────────

print("T5: Holonomy via force-extracted gauge")

# Extract 2x2 gauge matrix R of a z-bond from actual force computations.
# Probe with e1=(1,0) and e2=(0,1) at iz_hi, read correction at iz_lo.
# delta_f = K1*(R-I)*e_col, so R = (delta_f/K1) + I.

alpha5 = 0.37  # generic irrational value
force_fn5 = make_vortex_force(alpha5, R_LOOP, L, K1, K2)
iz_lo5 = CZ - 1
iz_hi5 = CZ

def extract_bond_gauge(force_fn, iy, ix):
    """Extract 2x2 gauge matrix R of z-bond at (iy, ix) from force."""
    R_ext = np.zeros((2, 2))
    for col, (dx, dy) in enumerate([(1.0, 0.0), (0.0, 1.0)]):
        ux_p = np.zeros((L, L, L))
        uy_p = np.zeros((L, L, L))
        uz_p = np.zeros((L, L, L))
        ux_p[iz_hi5, iy, ix] = dx
        uy_p[iz_hi5, iy, ix] = dy

        fv = force_fn(ux_p.copy(), uy_p.copy(), uz_p.copy())
        fp = scalar_laplacian_3d(ux_p, uy_p, uz_p, K1, K2)

        # (R-I) column
        R_ext[0, col] = (fv[0][iz_lo5, iy, ix] - fp[0][iz_lo5, iy, ix]) / K1
        R_ext[1, col] = (fv[1][iz_lo5, iy, ix] - fp[1][iz_lo5, iy, ix]) / K1
    # Add I back: R = (R-I) + I
    R_ext[0, 0] += 1.0
    R_ext[1, 1] += 1.0
    return R_ext

phi5 = 2 * np.pi * alpha5
R_expected = np.array([[np.cos(phi5), -np.sin(phi5)],
                        [np.sin(phi5),  np.cos(phi5)]])
I_mat = np.eye(2)

# Bond at center (inside disk)
R_center = extract_bond_gauge(force_fn5, CY, CX)
report("center bond gauge = R(2*pi*alpha)",
       np.max(np.abs(R_center - R_expected)) < 1e-14,
       f"max diff = {np.max(np.abs(R_center - R_expected)):.2e}")

# Bond at off-center position (inside disk: 3^2+2^2=13 < 25)
R_off = extract_bond_gauge(force_fn5, CY + 3, CX + 2)
report("off-center bond = same R",
       np.max(np.abs(R_off - R_expected)) < 1e-14,
       f"max diff = {np.max(np.abs(R_off - R_expected)):.2e}")

# Bond outside disk
ix_out5 = CX + int(R_LOOP) + 5
R_out = extract_bond_gauge(force_fn5, CY, ix_out5)
report("outside bond gauge = I",
       np.max(np.abs(R_out - I_mat)) < 1e-14,
       f"max diff = {np.max(np.abs(R_out - I_mat)):.2e}")

# Threading loop: up inside (R_center), down outside (R_out^T) -> holonomy
hol_thread = R_out.T @ R_center  # I^T @ R = R for outside bond = I
report("threading holonomy = R(2*pi*alpha)",
       np.max(np.abs(hol_thread - R_expected)) < 1e-14)

# Non-threading loop: both crossings inside disk -> R^T @ R = I
hol_double = R_off.T @ R_center
report("double-crossing holonomy = I",
       np.max(np.abs(hol_double - I_mat)) < 1e-14,
       f"max diff = {np.max(np.abs(hol_double - I_mat)):.2e}")

# Snap test: alpha=0.5 should give R = -I exactly (not ~1e-16 off)
force_fn_half = make_vortex_force(0.5, R_LOOP, L, K1, K2)
R_half = extract_bond_gauge(force_fn_half, CY, CX)
report("alpha=0.5: R = -I exact (snap)",
       np.max(np.abs(R_half - (-I_mat))) == 0.0,
       f"max diff = {np.max(np.abs(R_half - (-I_mat))):.2e}")


# ── T6: Self-adjoint at alpha != 0 ────────────────────────────

print("T6: Self-adjoint at alpha != 0")

# <u, F(v)> == <v, F(u)> for the full vortex operator at generic alpha.
# Analytic proof: delta_L is self-adjoint because u·Rv = (R^Tu)·v.
# Verify numerically at alpha=0.3 and alpha=0.5.

for alpha6 in [0.3, 0.5]:
    force_fn6 = make_vortex_force(alpha6, R_LOOP, L, K1, K2)
    rng6 = np.random.RandomState(123)
    u6 = [rng6.randn(L, L, L) for _ in range(3)]
    v6 = [rng6.randn(L, L, L) for _ in range(3)]

    fu = force_fn6(u6[0].copy(), u6[1].copy(), u6[2].copy())
    fv = force_fn6(v6[0].copy(), v6[1].copy(), v6[2].copy())

    # <u, F(v)> = sum_i u_i * Fv_i
    uFv = sum(np.sum(u6[i] * fv[i]) for i in range(3))
    vFu = sum(np.sum(v6[i] * fu[i]) for i in range(3))

    diff6 = abs(uFv - vFu)
    scale6 = max(abs(uFv), abs(vFu))
    rel6 = diff6 / scale6 if scale6 > 0 else diff6
    report(f"alpha={alpha6}: <u,Fv>==<v,Fu>", rel6 < 1e-12,
           f"rel diff = {rel6:.2e}")


# ── T7: Gauge invariance (Dirac surface independence) ────────

print("T7: Gauge invariance — move Dirac disk")

# Moving disk from cz to cz+1 is a gauge transform g(cz) = R at sites
# inside the disk. u'(cz) = R*u(cz), all else unchanged.
#
# Gauge invariance is APPROXIMATE on the lattice because the gauge
# transform at the disk boundary creates mismatched xy-bonds:
# site (cz, iy, ix) [inside, rotated] ↔ (cz, iy', ix') [outside, not rotated].
# These NN xy-bonds are never gauged → violation ~ O(perimeter / L^3).
#
# With K2>0, additional violation from NNN cross-layer bonds.
# Both violations are small and localized to the disk perimeter.

alpha7 = 0.3
phi7 = 2 * np.pi * alpha7
c7, s7 = np.cos(phi7), np.sin(phi7)

rng7 = np.random.RandomState(77)
ux7 = rng7.randn(L, L, L)
uy7 = rng7.randn(L, L, L)
uz7 = rng7.randn(L, L, L)

cz1 = CZ
cz2 = CZ + 1

# Gauge transform: rotate (ux, uy) at layer cz1, only inside disk
ux7g = ux7.copy()
uy7g = uy7.copy()
uz7g = uz7.copy()
ux7g[cz1, iy_disk, ix_disk] = c7 * ux7[cz1, iy_disk, ix_disk] - s7 * uy7[cz1, iy_disk, ix_disk]
uy7g[cz1, iy_disk, ix_disk] = s7 * ux7[cz1, iy_disk, ix_disk] + c7 * uy7[cz1, iy_disk, ix_disk]

def compute_energy(force_fn, ux, uy, uz):
    fv = force_fn(ux.copy(), uy.copy(), uz.copy())
    return -(np.sum(ux * fv[0]) + np.sum(uy * fv[1]) + np.sum(uz * fv[2]))

# K2=0 (NN only): violation from boundary xy-bonds only
f1_nn = make_vortex_force(alpha7, R_LOOP, L, K1, 0.0, cz=cz1)
f2_nn = make_vortex_force(alpha7, R_LOOP, L, K1, 0.0, cz=cz2)
E1_nn = compute_energy(f1_nn, ux7, uy7, uz7)
E2_nn = compute_energy(f2_nn, ux7g, uy7g, uz7g)
rel_nn = abs(E1_nn - E2_nn) / max(abs(E1_nn), abs(E2_nn))
report("K2=0: violation from boundary xy-bonds", rel_nn < 1e-3,
       f"rel diff = {rel_nn:.2e}")

# K2=0.5 (NN+NNN): additional violation from NNN cross-layer bonds
f1_full = make_vortex_force(alpha7, R_LOOP, L, K1, K2, cz=cz1)
f2_full = make_vortex_force(alpha7, R_LOOP, L, K1, K2, cz=cz2)
E1_full = compute_energy(f1_full, ux7, uy7, uz7)
E2_full = compute_energy(f2_full, ux7g, uy7g, uz7g)
rel_full = abs(E1_full - E2_full) / max(abs(E1_full), abs(E2_full))
report("K2=0.5: violation with NNN", rel_full < 1e-3,
       f"rel diff = {rel_full:.2e}")

# Both violations should be small (perimeter / volume ~ 30/27000 ~ 10^-3)
# and NNN should add to the NN violation
report("NNN adds to violation", rel_full >= rel_nn,
       f"K2=0: {rel_nn:.2e}, K2=0.5: {rel_full:.2e}")


# ── Summary ──────────────────────────────────────────────────

print()
print(f"Phase 2 gauge_3d: {passed}/{passed+failed} PASS")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
