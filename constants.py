"""Physical constants and the piecewise-exponential atmosphere table.

All distances in km, times in s, masses in kg, densities in kg/m^3.
"""

import numpy as np

# Earth gravity and shape
MU_EARTH = 398600.4418          # km^3 / s^2
R_EARTH = 6378.137              # km (equatorial)
J2 = 1.0826267e-3               # Earth second zonal harmonic
OMEGA_EARTH = 7.2921159e-5      # rad/s (sidereal rotation rate)

# Time conversions
DAY_TO_SEC = 86400.0
MIN_TO_SEC = 60.0

# Piecewise-exponential atmosphere (Vallado 4e, Table 8-4; 0-1000 km).
# For altitude h in km lying in bin [h_i, h_{i+1}):
#     rho(h) = RHO0[i] * exp( -(h - H0[i]) / H_SCALE[i] )
# Above 1000 km density is treated as zero.
_H0 = np.array([
      0.0,  25.0,  30.0,  40.0,  50.0,  60.0,  70.0,  80.0,  90.0, 100.0,
    110.0, 120.0, 130.0, 140.0, 150.0, 180.0, 200.0, 250.0, 300.0, 350.0,
    400.0, 450.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0,
])

_RHO0 = np.array([
    1.225,       3.899e-2,   1.774e-2,   3.972e-3,   1.057e-3,
    3.206e-4,    8.770e-5,   1.905e-5,   3.396e-6,   5.297e-7,
    9.661e-8,    2.438e-8,   8.484e-9,   3.845e-9,   2.070e-9,
    5.464e-10,   2.789e-10,  7.248e-11,  2.418e-11,  9.518e-12,
    3.725e-12,   1.585e-12,  6.967e-13,  1.454e-13,  3.614e-14,
    1.170e-14,   5.245e-15,  3.019e-15,
])

_H_SCALE = np.array([
      7.249,   6.349,   6.682,   7.554,   8.382,   7.714,   6.549,   5.799,
      5.382,   5.877,   7.263,   9.473,  12.636,  16.149,  22.523,  29.740,
     37.105,  45.546,  53.628,  53.298,  58.515,  60.828,  63.822,  71.835,
     88.667, 124.64,  181.05,  268.00,
])

ATMOSPHERE_H0 = _H0
ATMOSPHERE_RHO0 = _RHO0
ATMOSPHERE_H_SCALE = _H_SCALE
ATMOSPHERE_MAX_ALT = 1000.0     # km; density is zero above this
