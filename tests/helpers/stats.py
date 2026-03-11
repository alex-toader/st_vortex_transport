"""Statistical utilities: CV, power law fits."""
import numpy as np


def cv(arr):
    """Coefficient of variation (%) = std/|mean| * 100. Sign-blind (uses abs)."""
    arr = np.asarray(arr)
    return np.std(arr) / np.abs(np.mean(arr)) * 100


def N_eff_from_sigma(sigma_ring, sigma_bond):
    """N_eff = sigma_ring / sigma_bond."""
    return sigma_ring / sigma_bond


def log_log_slope(x, y):
    """Power law exponent from log-log linear fit. Returns (slope, intercept)."""
    lx = np.log(np.asarray(x))
    ly = np.log(np.asarray(y))
    return np.polyfit(lx, ly, 1)


# ── Self-consistency ─────────────────────────────────────────────
assert cv(np.array([5.0, 5.0, 5.0])) == 0.0, "CV of constant must be 0"
assert abs(cv(np.array([1.0, 3.0])) - 50.0) < 1e-10, "CV([1,3]) must be 50%"
# log_log_slope of y = x^2 should give slope 2
_slope, _ = log_log_slope(np.array([1.0, 2.0, 4.0]), np.array([1.0, 4.0, 16.0]))
assert abs(_slope - 2.0) < 1e-10, f"log_log_slope of x^2 gave {_slope}, expected 2"
