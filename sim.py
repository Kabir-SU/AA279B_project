"""User-facing API: Satellite, walker(), Simulator."""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from astro import coe_to_cart
from constants import MU_EARTH
from dynamics import accelerations


@dataclass
class Satellite:
    """ECI state (km, km/s) plus ballistic properties for a single spacecraft."""
    r: np.ndarray
    v: np.ndarray
    mass: float
    area: float
    cd: float = 2.2
    name: str = ""

    def __post_init__(self):
        self.r = np.asarray(self.r, dtype=float)
        self.v = np.asarray(self.v, dtype=float)

    @classmethod
    def from_elements(cls, elements, mass, area, cd=2.2, name=""):
        """Build from classical elements (a, e, inc, raan, arg_peri, M). Angles in rad."""
        r, v = coe_to_cart(elements, MU_EARTH)
        return cls(r=r, v=v, mass=mass, area=area, cd=cd, name=name)


def walker(T, P, F, a, e, inc_deg, mass, area, cd=2.2, arg_peri=0.0, raan0=0.0):
    """Walker T/P/F constellation builder.

    T : total number of satellites (must be divisible by P)
    P : number of equally-spaced orbital planes
    F : phase offset parameter (integer, 0 <= F < P)
    a : semi-major axis [km]
    e : eccentricity
    inc_deg : inclination [deg]
    mass, area, cd : per-satellite ballistic properties (applied uniformly)
    arg_peri : argument of periapsis [rad] (default 0; irrelevant for circular)
    raan0 : RAAN offset for plane 0 [rad]

    Returns list[Satellite] of length T, ordered plane-major then satellite-in-plane.
    """
    assert T % P == 0, f"T={T} must be divisible by P={P}"
    S = T // P
    inc = np.deg2rad(inc_deg)
    sats = []
    for p in range(P):
        raan = raan0 + 2.0 * np.pi * p / P
        for s in range(S):
            M = 2.0 * np.pi * (s / S + p * F / T)
            sats.append(Satellite.from_elements(
                elements=(a, e, inc, raan, arg_peri, M),
                mass=mass, area=area, cd=cd,
                name=f"w{p}_{s}",
            ))
    return sats


class Simulator:
    """Propagate a list of satellites forward in time under central-body + perturbations.

    All N satellites are integrated in a single solve_ivp call for efficiency.
    """

    def __init__(self, satellites, perturbations=("J2", "drag"),
                 earth_rotation=True, rtol=1e-12, atol=1e-12, method="DOP853"):
        self.satellites = list(satellites)
        self.perturbations = set(perturbations)
        self.earth_rotation = earth_rotation
        self.rtol = rtol
        self.atol = atol
        self.method = method

    def propagate(self, t_span, t_eval=None):
        """Integrate from t_span[0] to t_span[1].

        Returns
        -------
        times : ndarray (M,)
        states : ndarray (N, 6, M)   ECI (r, v) in km, km/s
        """
        N = len(self.satellites)
        flat0 = np.empty(6 * N)
        cda_over_m = np.empty(N)
        for i, sat in enumerate(self.satellites):
            flat0[6 * i:6 * i + 3] = sat.r
            flat0[6 * i + 3:6 * i + 6] = sat.v
            cda_over_m[i] = sat.cd * sat.area / sat.mass

        use_j2 = "J2" in self.perturbations
        use_drag = "drag" in self.perturbations

        sol = solve_ivp(
            accelerations,
            t_span,
            flat0,
            t_eval=t_eval,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            args=(cda_over_m, use_j2, use_drag, self.earth_rotation),
        )
        if not sol.success:
            raise RuntimeError(f"Propagation failed: {sol.message}")

        states = sol.y.reshape(N, 6, -1)
        return sol.t, states
