import numpy as np
import pandas as pd
import math
import os

def RK4_air2stream(Ta_series, Q_series, p, n_tot, Tice_cover=0.0):
    Tw = np.zeros(n_tot)
    Qmedia = np.mean(Q_series)
    TTT = 1.0 / 365.0

    p1, p2, p3, p4, p5, p6, p7, p8 = p[1:9]

    for j in range(n_tot - 1):
        Ta_j = Ta_series[j]
        Ta_j1 = Ta_series[j+1]
        Q_j = Q_series[j]
        Q_j1 = Q_series[j+1]
        Tw_j = Tw[j]
        tt_j = j * TTT

        Ta_mid = 0.5 * (Ta_j + Ta_j1)
        Q_mid = 0.5 * (Q_j + Q_j1)
        tt_mid = tt_j + 0.5 * TTT

        def RK(Ta, QQ, Tw_val, time):
            theta = QQ / Qmedia
            DD = theta ** p4
            return (p1 + p2 * Ta - p3 * Tw_val + theta * (p5 + p6 * math.cos(2.0 * math.pi * (time - p7)) - p8 * Tw_val)) / DD

        K1 = RK(Ta_j, Q_j, Tw_j, tt_j)
        K2 = RK(Ta_mid, Q_mid, Tw_j + 0.5 * K1, tt_mid)
        K3 = RK(Ta_mid, Q_mid, Tw_j + 0.5 * K2, tt_mid)
        K4 = RK(Ta_j1, Q_j1, Tw_j + K3, tt_j + TTT)

        Tw_j1 = Tw_j + (1.0 / 6.0) * (K1 + 2.0*K2 + 2.0*K3 + K4)
        if Tw_j1 < Tice_cover:
            Tw_j1 = Tice_cover
        Tw[j+1] = Tw_j1

    return Tw

def create_synthetic_data(years, out_file, is_future=False):
    n_days = 365 * years
    np.random.seed(42 if not is_future else 99)

    dates = pd.date_range(start="2020-01-01" if not is_future else f"{2020+years}-01-01", periods=n_days, freq='D')
    tt = np.arange(n_days) / 365.0

    T_air = 15.0 + 10.0 * np.sin(2.0 * math.pi * tt - math.pi/2) + np.random.normal(0, 3.0, n_days)
    Q = 50.0 + 30.0 * np.cos(2.0 * math.pi * tt - math.pi/4) + np.random.lognormal(0, 10.0, n_days)
    Q = np.clip(Q, 5.0, 200.0)

    p_true = np.zeros(9)
    p_true[1:9] = [0.1, 0.05, 0.05, 0.2, 1.0, 0.5, 0.5, 0.1]

    T_water = RK4_air2stream(T_air, Q, p_true, n_days)

    # Add autocorrelated noise to historical data to showcase AR(1) benefit
    if not is_future:
        sigma = 0.8
        rho = 0.6
        eps = np.random.standard_normal(n_days)
        epsilon = np.empty(n_days)
        epsilon[0] = sigma * eps[0]
        epsilon[1:] = sigma * np.sqrt(1 - rho**2) * eps[1:]
        import scipy.signal
        noise = scipy.signal.lfilter([1.0], [1.0, -rho], epsilon)
        T_water += noise
    else:
        T_water[:] = -999.0 # We don't know the future!

    df = pd.DataFrame({
        'Date': dates.strftime('%m/%d/%Y'),
        'T_air': np.round(T_air, 2),
        'T_water': np.round(T_water, 2),
        'Discharge': np.round(Q, 2)
    })

    df.to_csv(out_file, index=False)
    print(f"Generated {out_file}")

if __name__ == "__main__":
    os.makedirs('examples/forward_prediction_intervals', exist_ok=True)
    create_synthetic_data(3, 'examples/forward_prediction_intervals/historical_data.csv', is_future=False)
    create_synthetic_data(1, 'examples/forward_prediction_intervals/future_data.csv', is_future=True)
