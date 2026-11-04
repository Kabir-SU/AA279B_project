import numpy as np

from astro import coe_to_cart, cart_to_coe

from constants import (
    R_EARTH, J2, OMEGA_EARTH, MU_EARTH, G0
)

# def j2_coes_secular_rates(a, e, i_rad):
#     """(a, e, i_rad) -> (d_raan, d_aop, d_M0) in rad/s"""
#     p = a*(1 - e*e)
#     coeff = 0.75 * J2 * OMEGA_EARTH * R_EARTH**2 / (p**2)
#     d_raan = -2 * coeff * np.cos(i_rad)
#     d_aop = coeff * (4 - 5*np.sin(i_rad)**2)
#     d_M0 = -coeff * np.sqrt(1 - e*e) * (3*np.sin(i_rad)**2 - 2)
#     return d_raan, d_aop, d_M0

def algorithm_1(
    t_i, a_i, a_f, i_i, i_f, Omega_i, m_i, I_sp,
    T_max, rho_0, H_0, C_D, S, N, w_ecl,
    mu=MU_EARTH, R_e=R_EARTH,
    J2=J2,
):
    """Minimum-time two-orbital-element trageting (a_i,i_i) -> (a_f,i_f)

    Parameters
    ----------
    t_i : float
        Initial epoch [s].
    a_i, a_f : float
        Initial and final semimajor axes [km].
    i_i, i_f : float
        Initial and final inclinations [rad].
    Omega_i : float
        Initial right ascension of the ascending node (RAAN) [rad].
    m_i : float
        Initial spacecraft mass [kg].
    I_sp : float
        Specific impulse [s].
    T_max : float
        Maximum thrust [N].
    rho_0 : float
        Atmospheric density normalization in rho = rho_0*exp(-altitude/H_0)
        [kg/km^3]. Set to zero to neglect drag.
    H_0 : float
        Atmospheric scale height [km].
    C_D : float
        Dimensionless drag coefficient.
    S : float
        Reference drag area [km^2].
    N : int
        Number of transfer discretization points; at least two.
    w_ecl : callable
        Function (t, a, i, Omega) returning the orbit-averaged fraction
        available for thrusting, in (0, 1]. See [this](https://commons.erau.edu/ijaaa/vol6/iss5/15/)
    mu : float, optional
        Central-body gravitational parameter [km^3/s^2].
    R_e : float, optional
        Central-body equatorial radius [km].

    Returns
    -------
    Delta_V_f : float
        Transfer velocity increment [km/s].
    Delta_t : float
        Transfer duration, including eclipse and drag effects [s].
    Delta_Omega : float
        Accumulated, unwrapped J2 RAAN change [rad].
    """

    # 1
    V_i = np.sqrt(mu / a_i)
    V_f = np.sqrt(mu / a_f)
    Delta_i = i_f - i_i
    f_t = T_max / m_i

    # 2
    Delta_V_f = np.sqrt(
        V_i**2 + V_f**2
        - 2 * V_i * V_f * np.cos(np.pi / 2 * Delta_i)
    )

    # 3
    beta_0 = np.arctan2(
        np.sin(np.pi / 2 * Delta_i),
        (V_i / V_f - np.cos(np.pi / 2 * Delta_i))
    )

    # 4
    t_f = Delta_V_f / f_t + t_i
    Delta_t = (t_f - t_i) / (N - 1)

    t = np.zeros(N)
    Delta_V = np.zeros(N)
    a = np.zeros(N)
    V = np.zeros(N)
    i = np.zeros(N)
    beta = np.zeros(N)
    Omega = np.zeros(N)
    m = np.zeros(N)

    # 5–12
    for k in range(N):
        t[k] = t_i + k * Delta_t                         # 6
        Delta_V[k] = (t[k] - t_i) * f_t                 # 7

        a[k] = mu / (                                  # 8
            V_i**2 + Delta_V[k]**2
            - 2 * V_i * Delta_V[k] * np.cos(beta_0)
        )

        V[k] = np.sqrt(mu / a[k])                       # 9

        i[k] = i_i + 2 * np.sign(Delta_i) / np.pi * (    # 10
            np.arctan2(
                (Delta_V[k] - V_i * np.cos(beta_0)),
                (V_i * np.sin(beta_0))
            )
            + np.pi / 2 - beta_0
        )

        beta[k] = np.arctan2(                           # 11
            V_i * np.sin(beta_0),
            (V_i * np.cos(beta_0) - Delta_V[k])
        )

    # 13
    Omega[0] = Omega_i
    m[0] = m_i

    # 14–23
    for j in range(N - 1):
        w = w_ecl(t[j], a[j], i[j], Omega[j])            # 15

        m[j + 1] = m[j] * np.exp(                       # 16
            -(Delta_V[j + 1] - Delta_V[j]) / (I_sp * G0)
        )

        m_bar = (m[j] + m[j + 1]) / 2                   # 17

        f_d = S * rho_0 * C_D / 2 * 0.5 * (            # 18
            V[j]**2 / m[j] * np.exp(-(a[j] - R_e) / H_0)
            + V[j + 1]**2 / m[j + 1]
            * np.exp(-(a[j + 1] - R_e) / H_0)
        )

        f_t_j = T_max / m_bar - f_d * np.cos(           # 19
            (beta[j] + beta[j + 1]) / 2
        )

        t[j + 1] = t[j] + (                            # 20
            (Delta_V[j + 1] - Delta_V[j]) / (f_t_j * w)
        )

        Omega_dot_j = (                                # 21
            -3 / 2 * J2 * np.sqrt(mu / a[j]**3)
            * (R_e / a[j])**2 * np.cos(i[j])
        )

        Omega[j + 1] = Omega[j] + Omega_dot_j * (        # 22
            t[j + 1] - t[j]
        )

    return Delta_V_f, t[-1] - t_i, Omega[-1] - Omega_i


def algorithm_2(
    t_i, a_i, a_f, i_i, i_f, Omega_i, m_i, I_sp,
    T_max, rho_0, H_0, C_D, S, N,
    Omega_c_i, a_D, i_D, w_ecl,
    mu=MU_EARTH, R_e=R_EARTH,
    J2=J2
):
    """Minimum-time two-orbital-element targeting (a_i,i_i) -> (a_f,i_f)

    Parameters
    ----------
    t_i : float
        Initial epoch [s].
    a_i, a_f : float
        Initial and final semimajor axes [km].
    i_i, i_f : float
        Initial and final inclinations [rad].
    Omega_i : float
        Initial spacecraft RAAN [rad].
    m_i : float
        Initial spacecraft mass [kg].
    I_sp : float
        Specific impulse [s].
    T_max : float
        Maximum thrust [N].
    rho_0 : float
        Atmospheric density normalization in rho = rho_0*exp(-altitude/H_0)
        [kg/km^3]. Set to zero to neglect drag.
    H_0 : float
        Atmospheric scale height [km].
    C_D : float
        Dimensionless drag coefficient.
    S : float
        Reference drag area [km^2].
    N : int
        Number of discretization points per thrusting leg; at least two.
    Omega_c_i : float
        Target orbit RAAN at epoch t_i [rad].
    a_D : float
        Drift-orbit semimajor axis [km].
    i_D : float
        Drift-orbit inclination [rad].
    w_ecl : callable
        Function (t, a, i, Omega) returning the orbit-averaged fraction
        available for thrusting, in (0, 1].
    mu : float, optional
        Central-body gravitational parameter [km^3/s^2].
    R_e : float, optional
        Central-body equatorial radius [km].

    Returns
    -------
    Delta_V : float
        Total velocity increment, including drift drag compensation [km/s].
    ToF : float
        Total duration of the three phases [s].
    """
    # 1: Initial orbit -> drift orbit
    Delta_V_1, Delta_t_1, Delta_Omega_1 = algorithm_1(
        t_i, a_i, a_D, i_i, i_D, Omega_i, m_i, I_sp,
        T_max, rho_0, H_0, C_D, S, N, w_ecl,
        mu, R_e, J2
    )

    # 5: Computed early because phase 3 needs the updated mass
    m_D = m_i * np.exp(-Delta_V_1 / (I_sp * G0))

    # 2: Drift orbit -> final orbit
    Delta_V_3, Delta_t_3, Delta_Omega_3 = algorithm_1(
        t_i + Delta_t_1,
        a_D, a_f, i_D, i_f,
        Omega_i + Delta_Omega_1, m_D, I_sp,
        T_max, rho_0, H_0, C_D, S, N, w_ecl,
        mu, R_e, J2
    )

    Omega_dot_D = (
        -3 / 2 * J2 * np.sqrt(mu / a_D**3)
        * (R_e / a_D)**2 * np.cos(i_D)
    )
    Omega_dot_c = (
        -3 / 2 * J2 * np.sqrt(mu / a_f**3)
        * (R_e / a_f)**2 * np.cos(i_f)
    )

    # 3
    Delta_Omega_S = Omega_i + Delta_Omega_1 + Delta_Omega_3
    Delta_Omega_c = Omega_c_i + Omega_dot_c * (
        Delta_t_1 + Delta_t_3
    )

    # 4: Smallest nonnegative RAAN-matching drift duration
    rate = Omega_dot_D - Omega_dot_c
    gap = (Delta_Omega_c - Delta_Omega_S) % (2 * np.pi)

    if rate == 0:
        if not np.isclose(gap, 0):
            raise ValueError("No differential RAAN drift.")
        Delta_t_2 = 0.0
    elif rate > 0:
        Delta_t_2 = gap / rate
    else:
        Delta_t_2 = -((-gap) % (2 * np.pi)) / rate

    # 6
    f_d_2 = (
        S * rho_0 * C_D / 2
        * mu / (a_D * m_D)
        * np.exp(-(a_D - R_e) / H_0)
    )

    # 7: Constant drag acceleration approximation
    Delta_V_2 = f_d_2 * Delta_t_2

    Delta_V = Delta_V_1 + Delta_V_2 + Delta_V_3
    ToF = Delta_t_1 + Delta_t_2 + Delta_t_3

    return Delta_V, ToF