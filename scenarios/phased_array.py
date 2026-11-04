"""Bunch of sats in one orbit"""

import numpy as np

from sim import Simulator, walker, Satellite
from constants import DAY_TO_SEC, MIN_TO_SEC
from plots import summary

NAME = "walker_leo"


def run():
    phases = [0, 60, 120, 180, 240, 300]
    sats = [Satellite.from_elements(
        elements=(7000.0, 0.001, np.deg2rad(55.0), 0.0, 0.0, np.deg2rad(phase)),
        mass=100.0, area=1.0, cd=2.2,
    ) for phase in phases]

    t_span = 5 * MIN_TO_SEC

    sim = Simulator(sats, perturbations=("J2", "drag"))
    times, states = sim.propagate(
        (0.0, t_span),
        t_eval=np.linspace(0.0, t_span, 1000),
    )
    summary(times, states, sats, name=NAME, plot_endpoints=True)


if __name__ == "__main__":
    run()
