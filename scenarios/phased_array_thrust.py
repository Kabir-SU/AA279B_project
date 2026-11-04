"""Six satellites with continuous prograde tangential thrust for 60 minutes.

Run from the repository root: python -m scenarios.phased_array_thrust
"""

from pathlib import Path

import numpy as np

from constants import MIN_TO_SEC, POSITION_SLICE, VELOCITY_SLICE, MASS_INDEX
from plots import summary
from sim import Satellite, Simulator, Thruster

NAME = Path(__file__).stem
DURATION = 60 * MIN_TO_SEC
THROTTLE = 1.0


def tangential_thrust(t, state):
    """Thrust in the direction of motion with the radial component removed."""
    r = state[POSITION_SLICE]
    v = state[VELOCITY_SLICE]
    tangent = v - np.dot(v, r) / np.dot(r, r) * r
    return THROTTLE, tangent


def run(show=True):
    phases = [0, 60, 120, 180, 240, 300]
    thruster = Thruster(max_thrust=30, min_thrust=0.01, isp=1500.0)
    sats = [Satellite.from_elements(
        elements=(7000.0, 0.001, np.deg2rad(55.0), 0.0, 0.0, np.deg2rad(phase)),
        mass=100.0, area=1.0, cd=2.2, name=f"sat_{i}",
        dry_mass=90.0, thruster=thruster, thrust_command=tangential_thrust,
    ) for i, phase in enumerate(phases)]

    sim = Simulator(sats, perturbations=("J2", "drag"))
    times, states = sim.propagate(
        (0.0, DURATION), t_eval=np.linspace(0.0, DURATION, 1000), max_step=10.0,
    )
    summary(times, states, sats, name=NAME, show=show, plot_endpoints=True)
    for i, sat in enumerate(sats):
        used = sat.mass - states[i, MASS_INDEX, -1]
        print(f"{sat.name}: propellant used = {used:.6f} kg")
    return times, states


if __name__ == "__main__":
    run()
