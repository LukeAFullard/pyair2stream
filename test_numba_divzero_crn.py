from numba import njit
import numpy as np
import math

@njit
def fast_rk_version(version, p1, p2, p3, p4, p5, p6, p7, p8, Ta, QQ, Tw, time, Qmedia):
    theta = QQ / Qmedia

    # In air2stream, version 4 and 8 scale the energy input by a function of discharge.
    # The term is: dT/dt = (Energy_in) / (theta**p4)
    # When discharge drops to exactly zero, theta=0.
    # If p4 > 0, theta**p4 = 0, leading to division by zero and exploding temperatures.
    # Instead of making DD tiny, which explodes the term, if theta <= 0, we should assume
    # there is no flow, thus the advective/discharge-scaled terms vanish, or we should clamp DD to 1.0 (no scaling).
    # Actually, air2stream typically drops segments with zero flow entirely in gap-tolerant mode if Q<=0.
    return theta

print(fast_rk_version(8, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 5.0, 1.0, 100.0))
