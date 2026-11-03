"""Verification demos for the constellation propagator.

Run from the repo root:

    python -m example
"""

import time
import numpy as np

from sim import Satellite, Simulator, walker
from constants import MU_EARTH, R_EARTH, J2, DAY_TO_SEC
from astro import cart_to_coe


def hdr(name):
    print(f"\n=== {name} ===")


def two_body_closure():
    """One Keplerian period: position should close to near machine precision."""
    hdr("two-body closure")
    a = 7000.0
    T_period = 2.0 * np.pi * np.sqrt(a**3 / MU_EARTH)
    sat = Satellite.from_elements(
        elements=(a, 0.001, np.deg2rad(45.0), 0.3, 0.5, 0.7),
        mass=100.0, area=1.0, cd=2.2, name="kep",
    )
    sim = Simulator([sat], perturbations=())
    _, states = sim.propagate((0.0, T_period), t_eval=np.array([0.0, T_period]))
    err = np.linalg.norm(states[0, :3, -1] - states[0, :3, 0])
    print(f"|r(T) - r(0)| = {err:.2e} km   (target < 1e-6)")


def j2_nodal_regression():
    """ISS-like orbit; secular dRAAN/dt should match -3/2 n J2 (R/p)^2 cos i."""
    hdr("J2 nodal regression (ISS-like)")
    a = 6778.0
    e = 0.001
    inc = np.deg2rad(51.6)
    sat = Satellite.from_elements(
        elements=(a, e, inc, 0.0, 0.0, 0.0),
        mass=420000.0, area=2500.0, cd=2.2, name="iss",
    )
    sim = Simulator([sat], perturbations=("J2",))
    times, states = sim.propagate((0.0, DAY_TO_SEC),
                                  t_eval=np.linspace(0.0, DAY_TO_SEC, 500))
    raans = np.unwrap([
        cart_to_coe(states[0, :3, k], states[0, 3:, k], MU_EARTH)[3]
        for k in range(len(times))
    ])
    slope, _ = np.polyfit(times, raans, 1)
    n = np.sqrt(MU_EARTH / a**3)
    p = a * (1.0 - e**2)
    analytic = -1.5 * n * J2 * (R_EARTH / p) ** 2 * np.cos(inc)
    err_pct = 100.0 * abs(slope - analytic) / abs(analytic)
    print(f"numeric  dRAAN/dt = {np.rad2deg(slope) * DAY_TO_SEC:+.4f} deg/day")
    print(f"analytic dRAAN/dt = {np.rad2deg(analytic) * DAY_TO_SEC:+.4f} deg/day")
    print(f"error             = {err_pct:.3f}%   (target < 1%)")


def drag_decay():
    """ISS-like orbit; SMA averaged over an orbit should decrease each day."""
    hdr("drag decay (ISS-like, 400 km, J2 + drag)")
    a0 = R_EARTH + 400.0
    T_orbit = 2.0 * np.pi * np.sqrt(a0**3 / MU_EARTH)
    sat = Satellite.from_elements(
        elements=(a0, 0.001, np.deg2rad(51.6), 0.0, 0.0, 0.0),
        mass=420000.0, area=2500.0, cd=2.2, name="iss",
    )
    sim = Simulator([sat], perturbations=("J2", "drag"))
    times, states = sim.propagate((0.0, DAY_TO_SEC),
                                  t_eval=np.linspace(0.0, DAY_TO_SEC, 2000))
    smas = np.array([
        cart_to_coe(states[0, :3, k], states[0, 3:, k], MU_EARTH)[0]
        for k in range(len(times))
    ])
    # Average over first and last orbit to suppress J2 short-period oscillations
    sma_first = smas[times < T_orbit].mean()
    sma_last = smas[times > (times[-1] - T_orbit)].mean()
    print(f"<a> first orbit = {sma_first:.4f} km   (alt {sma_first - R_EARTH:.3f} km)")
    print(f"<a> last orbit  = {sma_last:.4f} km   (alt {sma_last - R_EARTH:.3f} km)")
    print(f"delta_<a>       = {(sma_last - sma_first) * 1000.0:+.1f} m "
          f"(expect negative, O(100 m/day) for ISS at 400 km)")


def walker_geometry():
    """Walker 24/6/2: 6 planes at 60 deg RAAN spacing, 30 deg phase offset between planes."""
    hdr("Walker 24/6/2 geometry")
    sats = walker(T=24, P=6, F=2, a=26559.0, e=0.0, inc_deg=55.0,
                  mass=1500.0, area=10.0, cd=2.2)
    elements = np.array([cart_to_coe(s.r, s.v, MU_EARTH) for s in sats])
    raans_deg = np.rad2deg(elements[:, 3]).reshape(6, 4)
    ms_deg = np.rad2deg(elements[:, 5]).reshape(6, 4)
    print("plane | RAAN [deg] | M of slot-0 [deg]")
    for p in range(6):
        print(f"  {p}   |  {raans_deg[p, 0]:6.2f}   |     {ms_deg[p, 0]:6.2f}")
    inter_plane = (ms_deg[1, 0] - ms_deg[0, 0]) % 360.0
    print(f"inter-plane phase offset = {inter_plane:.2f} deg   "
          f"(expected {2 * 360 / 24:.2f})")
    raan_spacing = (raans_deg[1, 0] - raans_deg[0, 0]) % 360.0
    print(f"RAAN spacing             = {raan_spacing:.2f} deg   "
          f"(expected {360 / 6:.2f})")


def vectorization_scaling():
    """Compare wall-clock for N=1 vs N=24 over one period."""
    hdr("vectorization scaling (1 vs 24 sats, 1 orbit)")

    def build(N):
        if N == 1:
            return [Satellite.from_elements(
                elements=(7000.0, 0.001, np.deg2rad(55.0), 0.0, 0.0, 0.0),
                mass=100.0, area=1.0, cd=2.2,
            )]
        return walker(T=N, P=6, F=1, a=7000.0, e=0.001, inc_deg=55.0,
                      mass=100.0, area=1.0, cd=2.2)

    def run(sats):
        sim = Simulator(sats, perturbations=("J2", "drag"))
        T_period = 2.0 * np.pi * np.sqrt(7000.0**3 / MU_EARTH)
        t0 = time.perf_counter()
        sim.propagate((0.0, T_period), t_eval=np.linspace(0.0, T_period, 100))
        return time.perf_counter() - t0

    # Warm up scipy / numpy caches
    run(build(1))

    t1 = run(build(1))
    t24 = run(build(24))
    print(f"N=1  : {t1 * 1e3:7.1f} ms")
    print(f"N=24 : {t24 * 1e3:7.1f} ms   (ratio {t24 / t1:.2f}x)")


if __name__ == "__main__":
    two_body_closure()
    j2_nodal_regression()
    drag_decay()
    walker_geometry()
    vectorization_scaling()
