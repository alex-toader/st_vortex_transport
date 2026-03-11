"""Box size convergence test: σ_tr at L=60, L=80, L=100.

Verifies that hardcoded FDTD data (L=80) is converged.
Uses 3 k-values (0.3, 0.9, 1.5) for speed.

Run: cd src && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 ../internal/box_convergence.py
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from parallel_fdtd import compute_references, compute_scattering

K1, K2 = 1.0, 0.5
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
ALPHA = 0.30
R = 5
N_WORKERS = 2

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)

k_test = np.array([0.3, 0.9, 1.5])

# Reference data from sigma_ring.py (L=80)
sigma_ref = {0.3: 40.740, 0.9: 4.852, 1.5: 1.754}

print("=" * 65)
print("Box Size Convergence: σ_tr at L=60, L=80, L=100")
print(f"R={R}, α={ALPHA}, DW={DW}, k={list(k_test)}")
print("=" * 65)

results = {}
for L in [60, 80, 100]:
    r_m = min(20, (L - 2 * DW) // 2 - 2)  # keep r_m inside non-PML zone
    print(f"\nL={L}, r_m={r_m}")
    t0 = time.time()

    refs = compute_references(k_test, L, DW, DS, DT, sx, r_m,
                              thetas, phis, K1, K2, n_workers=N_WORKERS)
    sigma = compute_scattering(k_test, refs, ALPHA, R, L, DW, DS, DT,
                               r_m, thetas, phis, K1, K2,
                               n_workers=N_WORKERS)

    elapsed = time.time() - t0
    results[L] = sigma
    print(f"  Time: {elapsed:.0f}s")
    for i, kv in enumerate(k_test):
        print(f"  k={kv}: σ_tr = {sigma[i]:.3f}")

print("\n" + "=" * 65)
print("Convergence summary")
print("=" * 65)
print(f"\n{'k':>4s}  {'L=60':>10s}  {'L=80':>10s}  {'L=100':>10s}  {'Δ(60-80)':>10s}  {'Δ(80-100)':>10s}")
for i, kv in enumerate(k_test):
    s60 = results[60][i]
    s80 = results[80][i]
    s100 = results[100][i]
    d1 = (s60 - s80) / s80 * 100
    d2 = (s80 - s100) / s100 * 100
    print(f"{kv:4.1f}  {s60:10.3f}  {s80:10.3f}  {s100:10.3f}  {d1:+9.1f}%  {d2:+9.1f}%")

# Also compare L=80 with hardcoded reference
print(f"\n{'k':>4s}  {'L=80 new':>10s}  {'hardcoded':>10s}  {'diff':>8s}")
for i, kv in enumerate(k_test):
    s_new = results[80][i]
    s_old = sigma_ref[kv]
    diff = (s_new - s_old) / s_old * 100
    print(f"{kv:4.1f}  {s_new:10.3f}  {s_old:10.3f}  {diff:+7.1f}%")
