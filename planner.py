import numpy as np

from constants import R_EARTH, J2, MU_EARTH, G0


def make_w_ecl(sun_vector_eci, R_e=R_EARTH):
    """Orbit-averaged sunlit fraction; sun_vector_eci(t) supplies the Sun direction.

    Use the same ECI frame for the Sun and orbit, km for radii, radians for
    angles, and define the calendar epoch inside sun_vector_eci.

    References
    ----------
    Sumanth R M - Computation of Eclipse Time for Low-Earth Orbiting Small Satellites Satellites 
    """
    def w_ecl(t, a, i, Omega):
        sun = np.asarray(sun_vector_eci(t), dtype=float)
        sun = sun / np.linalg.norm(sun)
        normal = np.array([np.sin(i)*np.sin(Omega),
                           -np.sin(i)*np.cos(Omega), np.cos(i)])
        sin_beta = np.clip(normal @ sun, -1.0, 1.0)
        ratio = R_e / a
        if abs(sin_beta) >= ratio:
            return 1.0
        argument = np.sqrt(1 - ratio**2) / np.sqrt(1 - sin_beta**2)
        return float(1 - np.arccos(np.clip(argument, 0, 1)) / np.pi)

    return w_ecl


def algorithm_1(
    t_i, a_i, a_f, i_i, i_f, Omega_i, m_i, I_sp,
    T_max, rho_0, H_0, C_D, S, N, w_ecl,
    mu=MU_EARTH, R_e=R_EARTH,
    J2=J2, *, return_details=False,
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

    return_details : bool, keyword-only
        Append a dictionary of phase/profile arrays without changing the calculation.

    Returns
    -------
    Delta_V_f : float
        Transfer velocity increment [km/s].
    Delta_t : float
        Transfer duration, including eclipse and drag effects [s].
    Delta_Omega : float
        Accumulated, unwrapped J2 RAAN change [rad].

    References
    ----------
    Riccardo - Low-Thrust Propulsion and Drift Orbit Optimization for Multi-Client Servicing Missions in Low Earth Orbit
    """

    if a_i == a_f and i_i == i_f:
        result = (0.0, 0.0, 0.0)
        return (*result, {"t": np.array([t_i]), "a": np.array([a_i]),
                          "i": np.array([i_i]), "Omega": np.array([Omega_i]),
                          "m": np.array([m_i]), "beta": np.array([0.0])}) if return_details else result

    # 1
    V_i = np.sqrt(mu / a_i)
    V_f = np.sqrt(mu / a_f)
    Delta_i = i_f - i_i
    f_t = T_max / m_i / 1000.0

    # 2
    Delta_V_f = np.sqrt(
        V_i**2 + V_f**2
        - 2 * V_i * V_f * np.cos(np.pi / 2 * Delta_i)
    )

    # 3: Use the steering magnitude; step 10 applies the inclination sign.
    beta_0 = np.arctan2(
        np.sin(np.pi / 2 * abs(Delta_i)),
        V_i / V_f - np.cos(np.pi / 2 * Delta_i),
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
            -(Delta_V[j + 1] - Delta_V[j]) / (I_sp * G0 / 1000.0)
        )

        m_bar = (m[j] + m[j + 1]) / 2                   # 17

        f_d = S * rho_0 * C_D / 2 * 0.5 * (            # 18
            V[j]**2 / m[j] * np.exp(-(a[j] - R_e) / H_0)
            + V[j + 1]**2 / m[j + 1]
            * np.exp(-(a[j + 1] - R_e) / H_0)
        )

        f_t_j = T_max / m_bar / 1000.0 - f_d * np.cos(           # 19
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

    result = (Delta_V_f, t[-1] - t_i, Omega[-1] - Omega_i)
    if return_details:
        return (*result, {"t": t, "a": a, "i": i, "Omega": Omega, "m": m, "beta": beta})
    return result


def algorithm_2(
    t_i, a_i, a_f, i_i, i_f, Omega_i, m_i, I_sp,
    T_max, rho_0, H_0, C_D, S, N,
    Omega_c_i, a_D, i_D, w_ecl,
    mu=MU_EARTH, R_e=R_EARTH, *, return_details=False,
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

    return_details : bool, keyword-only
        Append a dictionary of phase/profile arrays without changing the calculation.

    Returns
    -------
    Delta_V : float
        Total velocity increment, including drift drag compensation [km/s].
    ToF : float
        Total duration of the three phases [s].
    """
    # 1: Initial orbit -> drift orbit
    Delta_V_1, Delta_t_1, Delta_Omega_1, leg1 = algorithm_1(
        t_i, a_i, a_D, i_i, i_D, Omega_i, m_i, I_sp,
        T_max, rho_0, H_0, C_D, S, N, w_ecl,
        mu, R_e, J2, return_details=True,
    )

    # Mass after phase 1, needed for the second Algorithm 1 call.
    m_D = m_i * np.exp(-Delta_V_1 / (I_sp * G0 / 1000.0))

    # 2: Drift orbit -> final orbit (paper's non-iterative approximation).
    Delta_V_3, Delta_t_3, Delta_Omega_3, leg3 = algorithm_1(
        t_i + Delta_t_1, a_D, a_f, i_D, i_f,
        Omega_i + Delta_Omega_1, m_D, I_sp,
        T_max, rho_0, H_0, C_D, S, N, w_ecl,
        mu, R_e, J2, return_details=True,
    )

    Omega_dot_D = -1.5 * J2 * np.sqrt(mu/a_D**3) * (R_e/a_D)**2 * np.cos(i_D)
    Omega_dot_c = -1.5 * J2 * np.sqrt(mu/a_f**3) * (R_e/a_f)**2 * np.cos(i_f)

    # 3
    Delta_Omega_S = Omega_i + Delta_Omega_1 + Delta_Omega_3
    Delta_Omega_c = Omega_c_i + Omega_dot_c * (Delta_t_1 + Delta_t_3)

    # 4: Select the RAAN wrap in the direction of relative precession.
    rate = Omega_dot_D - Omega_dot_c
    gap = Delta_Omega_c - Delta_Omega_S
    Delta_t_2 = (gap % (2*np.pi))/rate if rate > 0 else -((-gap) % (2*np.pi))/rate

    # 5: m_D computed above.
    # 6
    f_d_2 = S * rho_0 * C_D / 2 * mu/(a_D*m_D) * np.exp(-(a_D-R_e)/H_0)

    # 7
    Delta_V_2 = f_d_2 * Delta_t_2
    Delta_V = Delta_V_1 + Delta_V_2 + Delta_V_3
    ToF = Delta_t_1 + Delta_t_2 + Delta_t_3
    if return_details:
        return Delta_V, ToF, {"leg1": leg1, "leg3": leg3,
                             "durations": (Delta_t_1, Delta_t_2, Delta_t_3)}
    return Delta_V, ToF
