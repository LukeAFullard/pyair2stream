import numpy as np

PI = np.pi

def RK4_air2stream(Ta, QQ, Tw, time, par, Qmedia):
    p = np.zeros(9)
    p[1:9] = par[0:8]
    DD = (QQ / Qmedia) ** p[4]
    K = p[1] + p[2]*Ta - p[3]*Tw + (QQ / Qmedia) * (p[5] + p[6]*np.cos(2.0*PI*(time - p[7])) - p[8]*Tw)
    K = K / DD
    return K

def test_physics_golden_match():
    # Synthetic inputs
    n_tot = 10
    par = np.array([1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1], dtype=np.float64)
    Qmedia = 10.0
    Tair = np.array([15.0]*n_tot, dtype=np.float64)
    Q = np.array([10.0]*n_tot, dtype=np.float64)
    tt = np.array([i/365.0 for i in range(n_tot)], dtype=np.float64)
    Twat_mod = np.zeros(n_tot, dtype=np.float64)
    Twat_mod[0] = 4.0
    TTT = 1.0/365.0
    Tice_cover = 0.0

    for j in range(n_tot - 1):
        K1 = RK4_air2stream(Tair[j], Q[j], Twat_mod[j], tt[j], par, Qmedia)
        K2 = RK4_air2stream(0.5 * (Tair[j] + Tair[j+1]), 0.5 * (Q[j] + Q[j+1]), Twat_mod[j] + 0.5 * K1, tt[j] + 0.5 * TTT, par, Qmedia)
        K3 = RK4_air2stream(0.5 * (Tair[j] + Tair[j+1]), 0.5 * (Q[j] + Q[j+1]), Twat_mod[j] + 0.5 * K2, tt[j] + 0.5 * TTT, par, Qmedia)
        K4 = RK4_air2stream(Tair[j+1], Q[j+1], Twat_mod[j] + K3, tt[j] + TTT, par, Qmedia)

        Twat_mod[j+1] = Twat_mod[j] + (1.0 / 6.0) * (K1 + 2.0*K2 + 2.0*K3 + K4)
        Twat_mod[j+1] = max(Twat_mod[j+1], Tice_cover)

    # The expected output from the original Fortran version
    expected = [
        4.0,
        5.540813708131667,
        6.802602297834198,
        7.836212193519514,
        8.683272897271095,
        9.377867630125097,
        9.94790114184635,
        10.416219582480279,
        10.801527378639673,
        11.119137910824824
    ]

    np.testing.assert_allclose(Twat_mod, expected, rtol=1e-9)
