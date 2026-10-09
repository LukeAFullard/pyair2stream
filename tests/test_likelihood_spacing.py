"""
The DE-MCMC likelihoods use how far apart the scored values are: errors d days apart have
correlation rho**d, so values on either side of a gap are related, but less than
consecutive ones. Every scored value counts (a weekly block whose middle day falls in a gap
was left out). DE-MCMC refuses KGE, whose best fit is not the least-squares fit its
uncertainty ranges are sampled around.
"""
import numpy as np
import pytest
import yaml

from pyair2stream.io import read_calibration
from pyair2stream.optimization import _ar1_log_likelihood
from pyair2stream.uncertainty import mean_error_variance_factor, scored_error_variance_factor


def _ar1(n, rho, rng, paths=1):
    x = np.empty((paths, n))
    x[:, 0] = rng.standard_normal(paths)
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(paths)
    return x


def test_the_factor_is_the_variance_of_the_mean_of_the_scored_errors():
    rng = np.random.default_rng(1)
    rho, n = 0.86, 1000
    # Consecutive days: close to (1 + rho) / (1 - rho) for a long record.
    assert scored_error_variance_factor(np.arange(2920), rho) == pytest.approx(mean_error_variance_factor(rho), rel=0.005)
    # Every other day: the correlation of neighbours is rho**2.
    every_other = np.arange(0, 2 * n, 2)
    assert scored_error_variance_factor(every_other, rho) == pytest.approx((1 + rho ** 2) / (1 - rho ** 2), rel=0.02)
    # Irregular measurement days: the factor matches simulated AR(1) errors.
    rows = np.sort(rng.choice(3 * n, n, replace=False))
    errors = _ar1(3 * n, rho, rng, paths=4000)[:, rows]
    measured = errors.mean(axis=1).var() * n          # variance of the mean, relative to independent errors
    assert scored_error_variance_factor(rows, rho) == pytest.approx(measured, rel=0.1)
    assert scored_error_variance_factor(rows, 0.0) == 1.0


def test_the_exact_likelihood_is_the_gaussian_density_of_the_scored_errors():
    # Up to a constant, -N/2 log(e' R^-1 e / N) - 1/2 log det R, with R_ij = rho**|t_i - t_j|.
    rng = np.random.default_rng(2)
    rho = 0.8
    rows = np.array([0, 1, 2, 5, 6, 9, 15, 16, 17, 30])
    R = rho ** np.abs(np.subtract.outer(rows, rows))
    _, logdet = np.linalg.slogdet(R)

    def direct(e):
        q = e[rows] @ np.linalg.solve(R, e[rows])
        return -0.5 * len(rows) * np.log(q / len(rows)) - 0.5 * logdet

    shifts = []
    for _ in range(3):
        e = rng.normal(size=31)
        shifts.append(_ar1_log_likelihood(e, rho, [rows]) - direct(e))
    assert np.allclose(shifts, shifts[0])
    # How the scored rows are split into runs does not matter: only their spacing.
    e = rng.normal(size=31)
    assert _ar1_log_likelihood(e, rho, [rows[:4], rows[4:]]) == pytest.approx(_ar1_log_likelihood(e, rho, [rows]))
    # Block means (weekly scoring) are treated as independent.
    weekly = np.arange(3, 300, 7)
    e = rng.normal(size=300)
    iid = -0.5 * len(weekly) * np.log(np.sum(e[weekly] ** 2) * (1 - rho ** 2) / len(weekly)) \
        + len(weekly) * 0.5 * np.log(1 - rho ** 2)
    assert _ar1_log_likelihood(e, rho, [weekly], block_days=7) == pytest.approx(iid)


def test_de_mcmc_refuses_kge(tmp_path):
    cfg = {"version": 3, "run_mode": "DE-MCMC", "objective_function": "KGE",
           "paths": {"input_data": "in.csv", "output_dir": str(tmp_path / "o")}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="DE-MCMC cannot be used with objective_function KGE"):
        read_calibration(str(tmp_path / "c.yaml"))
    for ok in ({"objective_function": "NSE"}, {"objective_function": "RMS"}, {"run_mode": "DE"}):
        (tmp_path / "c.yaml").write_text(yaml.safe_dump({**cfg, **ok}))
        read_calibration(str(tmp_path / "c.yaml"))
