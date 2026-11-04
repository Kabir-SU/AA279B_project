"""Walker 24/6/2 at ~700 km LEO, i=55 deg, propagated 1 day with J2 + drag. - super simple scenario"""

import numpy as np

from sim import Simulator, walker
from constants import DAY_TO_SEC
from plots import summary

NAME = "walker_leo"
# NAME = __file__.split("/")[-1].split(".")[0]


def run():
    sats = walker(T=24, P=6, F=2,
                  a=7078.137, e=0.0, inc_deg=55.0,
                  mass=500.0, area=5.0, cd=2.2)
    sim = Simulator(sats, perturbations=("J2", "drag"))
    times, states = sim.propagate(
        (0.0, DAY_TO_SEC),
        t_eval=np.linspace(0.0, DAY_TO_SEC, 1000),
    )
    summary(times, states, sats, name=NAME)


if __name__ == "__main__":
    run()
