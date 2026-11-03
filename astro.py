"""Orbital-element / Cartesian conversions and the small rotations they need.

Angles in radians, distances in km, speeds in km/s. Elements are classical
Keplerian: (a, e, inc, raan, arg_peri, M).
"""

import numpy as np


def wrap_to_2pi(angle):
    return angle % (2.0 * np.pi)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def solve_kepler(M, e, eps=1e-12, max_iter=50):
    """Newton-Raphson solve of Kepler's equation E - e sin E = M."""
    M = wrap_to_2pi(M)
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = f / fp
        E -= dE
        if abs(dE) < eps:
            break
    return wrap_to_2pi(E)


def rv_perifocal(a, e, E, mu):
    """Position and velocity in the perifocal (PQW) frame."""
    n = np.sqrt(mu / a**3)
    r = np.array([a * (np.cos(E) - e),
                  a * np.sqrt(1.0 - e**2) * np.sin(E),
                  0.0])
    v = (a * n / (1.0 - e * np.cos(E))) * np.array([
        -np.sin(E),
        np.sqrt(1.0 - e**2) * np.cos(E),
        0.0,
    ])
    return r, v


def rot_perifocal_to_eci(inc, raan, arg_peri):
    """3-1-3 rotation PQW -> ECI (Vallado 2-74)."""
    cO, sO = np.cos(raan), np.sin(raan)
    co, so = np.cos(arg_peri), np.sin(arg_peri)
    ci, si = np.cos(inc), np.sin(inc)
    return np.array([
        [cO * co - sO * so * ci, -cO * so - sO * co * ci,  sO * si],
        [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
        [so * si,                 co * si,                 ci     ],
    ])


def coe_to_cart(elements, mu):
    """Classical elements (a, e, inc, raan, arg_peri, M) -> ECI (r, v) in km, km/s."""
    a, e, inc, raan, arg_peri, M = elements
    E = solve_kepler(M, e)
    r_pqw, v_pqw = rv_perifocal(a, e, E, mu)
    R = rot_perifocal_to_eci(inc, raan, arg_peri)
    return R @ r_pqw, R @ v_pqw


def cart_to_coe(r_vec, v_vec, mu, tol=1e-10):
    """ECI (r, v) -> classical elements (a, e, inc, raan, arg_peri, M).

    Handles degenerate cases:
      - circular (e < tol): M becomes the argument of latitude
      - equatorial (|n_vec| < tol): raan is set to 0, arg_peri becomes longitude of periapsis
      - circular + equatorial: M becomes the true longitude
    """
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    n_vec = np.cross([0.0, 0.0, 1.0], h_vec)
    n = np.linalg.norm(n_vec)

    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    e = np.linalg.norm(e_vec)

    xi = 0.5 * v**2 - mu / r
    a = -mu / (2.0 * xi)

    inc = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

    circular = e < tol
    equatorial = n < tol

    if not equatorial:
        raan = np.arctan2(n_vec[1], n_vec[0])
    else:
        raan = 0.0

    if not circular and not equatorial:
        arg_peri = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0))
        if e_vec[2] < 0.0:
            arg_peri = 2.0 * np.pi - arg_peri
    elif not circular and equatorial:
        arg_peri = np.arctan2(e_vec[1], e_vec[0])
    else:
        arg_peri = 0.0

    if not circular:
        cos_nu = np.dot(e_vec, r_vec) / (e * r)
        nu = np.arccos(np.clip(cos_nu, -1.0, 1.0))
        if np.dot(r_vec, v_vec) < 0.0:
            nu = 2.0 * np.pi - nu
        E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(0.5 * nu),
                             np.sqrt(1.0 + e) * np.cos(0.5 * nu))
        M = E - e * np.sin(E)
    elif not equatorial:
        # Circular inclined: argument of latitude from the ascending node.
        cos_u = np.dot(n_vec, r_vec) / (n * r)
        u = np.arccos(np.clip(cos_u, -1.0, 1.0))
        if r_vec[2] < 0.0:
            u = 2.0 * np.pi - u
        M = u
    else:
        # Circular equatorial: true longitude from +x.
        M = np.arctan2(r_vec[1], r_vec[0])

    return (a, e, inc, wrap_to_2pi(raan), wrap_to_2pi(arg_peri), wrap_to_2pi(M))
