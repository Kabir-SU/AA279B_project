"""User-facing API: Satellite, walker(), Simulator."""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from astro import coe_to_cart
from constants import MU_EARTH, R_EARTH, N_STATES, POSITION_SLICE, VELOCITY_SLICE, MASS_INDEX
from dynamics import accelerations


@dataclass
class Thruster:
    max_thrust: float
    isp: float
    min_thrust: float = 0.0

    def __post_init__(self):
        if not (0 <= self.min_thrust <= self.max_thrust < np.inf
                and 0 < self.max_thrust and 0 < self.isp < np.inf):
            raise ValueError("Require 0 <= min_thrust <= max_thrust, max_thrust > 0, isp > 0.")

    def force(self, throttle):
        """Clamp to 0-1"""
        throttle = np.clip(throttle, 0, 1)
        force = throttle * self.max_thrust
        return force


@dataclass
class Satellite:
    """ECI state (km, km/s) plus ballistic properties for a single spacecraft."""
    r: np.ndarray
    v: np.ndarray
    mass: float
    area: float
    cd: float = 2.2
    name: str = ""
    thruster: Thruster = None
    dry_mass: float = None
    thrust_command: object = None

    def __post_init__(self):
        self.r = np.asarray(self.r, dtype=float)
        self.v = np.asarray(self.v, dtype=float)

    @classmethod
    def from_elements(cls, elements, mass, area, cd=2.2, name="", *,
                      thruster=None, dry_mass=None, thrust_command=None):
        """Build from classical elements (a, e, inc, raan, arg_peri, M). Angles in rad."""
        r, v = coe_to_cart(elements, MU_EARTH)
        return cls(r, v, mass, area, cd, name, thruster, dry_mass, thrust_command)


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

    def __init__(self, satellites: list[Satellite], perturbations=("J2", "drag"),
                 earth_rotation=True, rtol=1e-12, atol=1e-12, method="DOP853"):
        self.satellites = list(satellites)
        self.perturbations = set(perturbations)
        self.earth_rotation = earth_rotation
        self.rtol = rtol
        self.atol = atol
        self.method = method

    def propagate(self, t_span, t_eval=None, max_step=np.inf, *, min_altitude=None):
        """Integrate from t_span[0] to t_span[1].
        max_step limits adaptive steps [s]
        min_altitude optionally stops all satellites when any descends through
        that altitude [km]. The exact stopping state is included in the output.

        Returns
        -------
        times : ndarray (M,)
        states : ndarray (N, N_STATES, M)
            ECI position [km], velocity [km/s], and mass [kg].
        """
        N = len(self.satellites)
        initial_state = np.zeros((N, N_STATES))
        cda = np.empty(N)
        for i, sat in enumerate(self.satellites):
            if not np.isfinite(sat.mass) or sat.mass <= 0:
                raise ValueError("Satellite mass must be finite and positive.")
            initial_state[i, POSITION_SLICE] = sat.r
            initial_state[i, VELOCITY_SLICE] = sat.v
            initial_state[i, MASS_INDEX] = sat.mass
            cda[i] = sat.cd * sat.area

        use_j2 = "J2" in self.perturbations
        use_drag = "drag" in self.perturbations

        events = None
        if min_altitude is not None:
            def altitude_stop(t, flat_state, *args):
                positions = flat_state.reshape(N, N_STATES)[:, POSITION_SLICE]
                return np.min(np.linalg.norm(positions, axis=1)) - R_EARTH - min_altitude
            altitude_stop.terminal = True
            altitude_stop.direction = -1
            events = altitude_stop

        sol = solve_ivp(
            accelerations,
            t_span,
            initial_state.ravel(),
            t_eval=t_eval,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            max_step=max_step,
            events=events,
            args=(cda, use_j2, use_drag, self.earth_rotation, self.satellites),
        )
        if not sol.success:
            raise RuntimeError(f"Propagation failed: {sol.message}")

        times, values = sol.t, sol.y
        if events is not None and len(sol.t_events[0]):
            stop_time = sol.t_events[0][0]
            if len(times) == 0:
                values = np.empty((N * N_STATES, 0))
            if len(times) == 0 or times[-1] != stop_time:
                times = np.append(times, stop_time)
                values = np.column_stack([values, sol.y_events[0][0]])
        states = values.reshape(N, N_STATES, -1)

        # Clamp numerical undershoot at fuel cutoff.
        for i, sat in enumerate(self.satellites):
            if sat.thrust_command is not None:
                states[i, MASS_INDEX] = np.maximum(states[i, MASS_INDEX], sat.dry_mass)
        return times, states
