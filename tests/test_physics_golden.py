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
    # To properly match Fortran's behavior without running the entire call_model loop
    # we replicate Fortran's internal 365 day warmup here using just our raw physics function!
    n_tot_raw = 10
    n_tot = n_tot_raw + 365

    par = np.array([1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1], dtype=np.float64)
    Qmedia = 10.0
    Tair = np.array([15.0]*n_tot, dtype=np.float64)
    Q = np.array([10.0]*n_tot, dtype=np.float64)

    tt = np.zeros(n_tot, dtype=np.float64)
    k = 0
    for j in range(1, 366):
        tt[k + j - 1] = j / 365.0
    k = 365
    for j in range(1, 367):
        if k + j - 1 >= n_tot:
            break
        tt[k + j - 1] = j / 366.0

    Twat_mod = np.zeros(n_tot, dtype=np.float64)
    Twat_mod[0] = 4.0
    Tice_cover = 0.0

    for j in range(n_tot - 1):
        ttt = tt[j+1] - tt[j]
        # In Fortran, ttt is not simply difference if dates are irregular, but here they are sequential
        # Actually in Fortran, RK4 subroutines are called with tt(j) + 0.5*ttt etc
        # However, for pure sequential days, ttt is 1/365 or 1/366
        # Let's use difference! Wait, no, Fortran code sets ttt dynamically in some cases, but actually
        # for daily resolution ttt = 1.0/365.0 statically for the most part or based on leap years!
        # Actually, let's just use 1.0/365.0 for the warmup year and 1.0/366.0 for leap year.
        if j < 365:
            ttt = 1.0/365.0
        else:
            ttt = 1.0/366.0

        K1 = RK4_air2stream(Tair[j], Q[j], Twat_mod[j], tt[j], par, Qmedia)
        K2 = RK4_air2stream(0.5 * (Tair[j] + Tair[j+1]), 0.5 * (Q[j] + Q[j+1]), Twat_mod[j] + 0.5 * K1, tt[j] + 0.5 * ttt, par, Qmedia)
        K3 = RK4_air2stream(0.5 * (Tair[j] + Tair[j+1]), 0.5 * (Q[j] + Q[j+1]), Twat_mod[j] + 0.5 * K2, tt[j] + 0.5 * ttt, par, Qmedia)
        K4 = RK4_air2stream(Tair[j+1], Q[j+1], Twat_mod[j] + K3, tt[j] + ttt, par, Qmedia)

        Twat_mod[j+1] = Twat_mod[j] + (1.0 / 6.0) * (K1 + 2.0*K2 + 2.0*K3 + K4)
        Twat_mod[j+1] = max(Twat_mod[j+1], Tice_cover)

    from tests.fortran_runner import run_fortran_model

    expected = run_fortran_model(
        version=8,
        mod_num="RK4",
        n_tot_raw=n_tot_raw,
        Tair=Tair[365:],
        Q=Q[365:],
        par=par,
        Qmedia=Qmedia,
        Twat_initial=4.0
    )

    np.testing.assert_allclose(Twat_mod[365:], expected, rtol=1e-2, atol=1e-2)
