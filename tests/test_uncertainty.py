import numpy as np
import scipy.signal
from pyair2stream.uncertainty import estimate_ar1_rho, generate_ar1_noise, MIN_PAIRS_FOR_RHO_ESTIMATE

def test_estimate_ar1_rho_recovery():
    """Test that estimate_ar1_rho recovers a known rho from synthetic AR(1) data."""
    rng = np.random.default_rng(42)
    n = 1000

    # Generate true AR(1) residuals
    rho_true = 0.6
    sigma_true = 1.0

    eps = rng.standard_normal(n)
    epsilon = np.empty(n)
    epsilon[0] = sigma_true * eps[0]
    epsilon[1:] = sigma_true * np.sqrt(1 - rho_true**2) * eps[1:]

    residuals = scipy.signal.lfilter([1.0], [1.0, -rho_true], epsilon)

    # Set up dummy variables for the function
    Twat_mod = residuals
    Twat_obs = np.zeros(n) # residuals = Twat_mod - Twat_obs => Twat_mod
    eval_mask = np.ones(n, dtype=bool)
    segments = [(0, n - 1)]

    rho_est = estimate_ar1_rho(Twat_mod, Twat_obs, eval_mask, segments)

    # Should be close to the true rho
    assert np.isclose(rho_est, rho_true, atol=0.1)

def test_estimate_ar1_rho_segment_isolation():
    """Test that pairs are strictly excluded across segment boundaries."""
    n = 100
    residuals = np.zeros(n)
    # segment 1 (0..49)
    # segment 2 (50..99)
    # create artificial huge jump between 49 and 50 to test isolation

    rng = np.random.default_rng(42)

    rho_true = 0.0
    # True data has rho=0 (white noise)
    residuals[0:50] = rng.standard_normal(50)
    residuals[50:100] = rng.standard_normal(50) + 100.0 # Shift mean

    # But if we did cross boundaries, 49 and 50 would be highly correlated with a trend if we had many such jumps
    # Better test: positive correlation within segment, but alternating signs per segment
    # Let's just create a sequence where cross-segment pair would artificially inflate or deflate rho

    residuals = np.array([float(i % 2) for i in range(100)]) # perfectly negative corr (-1)

    # If we split into many small segments of size 2: [0, 1], [0, 1]
    # within segment: pairs are (0,1). Cross segment: pairs are (1,0)
    # Without isolation, we'd get pairs (0,1) and (1,0)

    segments = [(i, i+1) for i in range(0, 100, 2)]
    # All within-segment pairs are (0, 1). Wait, if residuals are 0,1,0,1,0,1
    # within segments (0,1), (2,3) etc., the pair is (val at 0, val at 1) -> (0, 1)

    # To test properly, let's use the code logic directly to ensure no cross-segment indices
    # We can use monkeypatching or just check it doesn't use the boundary

    # Let's make eval_mask true everywhere
    eval_mask = np.ones(100, dtype=bool)
    Twat_obs = np.zeros(100)
    Twat_mod = residuals

    rho_est = estimate_ar1_rho(Twat_mod, Twat_obs, eval_mask, segments)
    # In each segment [start, end] = [0, 1], pairs are: t=1: (resid[0], resid[1]) -> (0, 1)
    # So all pairs are (0, 1). Variance of first element is 0 (always 0), variance of second is 0 (always 1).
    # Correlation is undefined (NaN), which falls back to 0.0

    assert rho_est == 0.0

    # Let's do a simpler deterministic check:
    # Segments: (0, 40) and (60, 99)
    segments = [(0, 40), (60, 99)]
    Twat_obs = np.zeros(100)
    Twat_mod = np.arange(100, dtype=float) # slope=1, perfectly correlated (rho=1.0)

    # Without isolation, pair (40, 41) etc would be included. But segment is only up to 40.
    eval_mask = np.ones(100, dtype=bool)

    # Should be 0.99 (clipped)
    rho_est = estimate_ar1_rho(Twat_mod, Twat_obs, eval_mask, segments)
    assert rho_est == 0.99

def test_estimate_ar1_rho_gap_exclusion():
    """Test that missing obs (-999.0) and False eval_mask exclude pairs."""
    n = 50
    segments = [(0, n - 1)]
    Twat_mod = np.arange(n, dtype=float) # perfectly correlated
    Twat_obs = np.zeros(n)

    # Mask every other point
    eval_mask = np.ones(n, dtype=bool)
    eval_mask[1::2] = False

    # Now there are NO consecutive valid points
    rho_est = estimate_ar1_rho(Twat_mod, Twat_obs, eval_mask, segments)

    # Falls back to 0.0 due to < 30 pairs
    assert rho_est == 0.0

def test_estimate_ar1_rho_fallback():
    """Test fallback when < 30 pairs."""
    n = 29
    segments = [(0, n - 1)]
    Twat_mod = np.arange(n, dtype=float)
    Twat_obs = np.zeros(n)
    eval_mask = np.ones(n, dtype=bool)

    # n-1 = 28 pairs
    rho_est = estimate_ar1_rho(Twat_mod, Twat_obs, eval_mask, segments)
    assert rho_est == 0.0

    # 31 points -> 30 pairs -> should compute it
    n = 31
    segments = [(0, n - 1)]
    Twat_mod = np.arange(n, dtype=float)
    Twat_obs = np.zeros(n)
    eval_mask = np.ones(n, dtype=bool)
    rho_est = estimate_ar1_rho(Twat_mod, Twat_obs, eval_mask, segments)
    assert rho_est == 0.99 # Clipped

def test_generate_ar1_noise():
    """Test empirical variance, empirical lag-1 autocorrelation, and segment independence."""
    n_tot = 100000
    sigma = 2.0
    rho = 0.5
    segments = [(0, 49999), (50000, 99999)]
    rng = np.random.default_rng(42)

    noise = generate_ar1_noise(n_tot, sigma, rho, segments, rng)

    # Test variance (should be sigma^2 = 4.0)
    assert np.isclose(np.var(noise), sigma**2, rtol=0.05)

    # Test t=0 variance for the first segment
    # It's hard to test variance of a single point without many runs, but we can check it's not strictly 0
    # Let's generate 1000 times and check variance of first point
    first_points = []
    rng2 = np.random.default_rng(42)
    for _ in range(1000):
        n = generate_ar1_noise(10, sigma, rho, [(0, 9)], rng2)
        first_points.append(n[0])

    assert np.isclose(np.var(first_points), sigma**2, rtol=0.1)

    # Test lag-1 autocorrelation within segment
    seg1 = noise[0:50000]
    emp_rho = np.corrcoef(seg1[:-1], seg1[1:])[0, 1]
    assert np.isclose(emp_rho, rho, atol=0.05)

    # Test outside segments remain 0.0
    n_tot2 = 100
    segments2 = [(10, 20)]
    noise2 = generate_ar1_noise(n_tot2, sigma, rho, segments2, rng)

    assert np.all(noise2[:10] == 0.0)
    assert np.all(noise2[21:] == 0.0)
    assert not np.all(noise2[10:21] == 0.0)
