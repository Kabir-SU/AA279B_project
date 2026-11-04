"""Vectorized dynamics for N spacecraft about Earth.

flat_state has shape (N_STATES * N,), with each satellite storing
[rx, ry, rz, vx, vy, vz, mass] in km, km/s, and kg.
"""

import numpy as np

from constants import (
    MU_EARTH, R_EARTH, J2, OMEGA_EARTH, G0,
    N_STATES, POSITION_SLICE, VELOCITY_SLICE, MASS_INDEX,
    ATMOSPHERE_H0, ATMOSPHERE_RHO0, ATMOSPHERE_H_SCALE, ATMOSPHERE_MAX_ALT,
)


def density(h_km):
    """Piecewise-exponential atmospheric density [kg/m^3]. Zero above 1000 km."""
    h = np.asarray(h_km, dtype=float)
    idx = np.clip(
        np.searchsorted(ATMOSPHERE_H0, h, side="right") - 1,
        0, len(ATMOSPHERE_H0) - 1,
    )
    rho = ATMOSPHERE_RHO0[idx] * np.exp(
        -(h - ATMOSPHERE_H0[idx]) / ATMOSPHERE_H_SCALE[idx]
    )
    return np.where((h >= 0.0) & (h < ATMOSPHERE_MAX_ALT), rho, 0.0)


def accelerations(t, flat_state, cda, use_j2, use_drag, earth_rotation, satellites):
    """Right-hand side for N-satellite propagation. Returns d/dt flat_state."""
    N = cda.size
    state = flat_state.reshape(N, N_STATES)
    r = state[:, POSITION_SLICE]
    v = state[:, VELOCITY_SLICE]
    mass = state[:, MASS_INDEX].copy()
    for i, sat in enumerate(satellites):
        if sat.thrust_command is not None:
            mass[i] = max(mass[i], sat.dry_mass)

    r_norm = np.linalg.norm(r, axis=1)

    # Central-body gravity (always on)
    a = -MU_EARTH * r / (r_norm**3)[:, None]

    if use_j2:
        z_over_r_sq = (r[:, 2] / r_norm) ** 2
        k = -1.5 * J2 * MU_EARTH * R_EARTH**2 / r_norm**5
        xy_factor = 1.0 - 5.0 * z_over_r_sq
        z_factor = 3.0 - 5.0 * z_over_r_sq
        a[:, 0] += k * r[:, 0] * xy_factor
        a[:, 1] += k * r[:, 1] * xy_factor
        a[:, 2] += k * r[:, 2] * z_factor

    if use_drag:
        altitude = r_norm - R_EARTH
        rho = density(altitude)
        if earth_rotation:
            # omega x r = (0, 0, w) x (x, y, z) = (-w*y, w*x, 0)
            v_rel = v.copy()
            v_rel[:, 0] += OMEGA_EARTH * r[:, 1]
            v_rel[:, 1] -= OMEGA_EARTH * r[:, 0]
        else:
            v_rel = v
        v_rel_norm = np.linalg.norm(v_rel, axis=1)
        # a_SI[m/s^2] = -0.5 * rho[kg/m^3] * (Cd*A/m)[m^2/kg] * |v_SI| * v_SI
        # Convert to km/s^2 when v is in km/s: multiply by 1e3
        # (plugging km/s into v*v drops the result by 1e6, converting m/s^2 -> km/s^2 adds 1e-3,
        #  net factor 1e3).
        drag_coef = -500.0 * rho * (cda / mass) * v_rel_norm
        a += drag_coef[:, None] * v_rel

    deriv = np.zeros_like(state)
    for i, sat in enumerate(satellites):
        if sat.thrust_command is None or state[i, MASS_INDEX] <= sat.dry_mass:
            continue
        throttle, direction = sat.thrust_command(t, state[i].copy())
        force = sat.thruster.force(throttle)
        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if direction.shape != (3,) or not 0 < norm < np.inf:
            raise ValueError("Active thrust requires a finite, nonzero ECI direction vector.")
        a[i] += force / mass[i] * direction / norm / 1000.0
        deriv[i, MASS_INDEX] = -force / (sat.thruster.isp * G0)
    deriv[:, POSITION_SLICE] = v
    deriv[:, VELOCITY_SLICE] = a
    return deriv.ravel()
