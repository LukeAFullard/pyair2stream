import sys

filepath = 'tests/test_cross_validation.py'
with open(filepath, 'r') as f:
    content = f.read()

new_test = """
def test_run_leave_one_year_out_cv_gap_tolerant(dummy_data):
    from pyair2stream.cross_validation import run_leave_one_year_out_cv, summarize
    np.random.seed(42)

    dummy_data.gap_tolerant = True
    dummy_data.n_run = 10
    dummy_data.version = 5
    n_tot = dummy_data.n_tot
    dummy_data.Tair = np.sin(np.linspace(0, 4 * np.pi, n_tot)) + 10
    dummy_data.Twat_obs = np.sin(np.linspace(0, 4 * np.pi, n_tot)) * 0.8 + 10
    dummy_data.Q = np.ones(n_tot) * 10
    dummy_data.tt = np.linspace(0, 4, n_tot)

    dummy_data.parmin = np.zeros(8)
    dummy_data.parmax = np.ones(8) * 10
    dummy_data.par = np.ones(8)
    dummy_data.par_best = np.ones(8)
    dummy_data.flag_par = np.ones(8, dtype=bool)

    # Initialize required arrays
    dummy_data.Twat_mod = np.zeros(n_tot)
    dummy_data.Twat_mod_agg = np.zeros(n_tot)
    dummy_data.Twat_obs_agg = np.zeros(n_tot)
    dummy_data.I_pos = np.zeros(n_tot, dtype=np.int32)
    dummy_data.I_inf = np.zeros(n_tot, dtype=np.int32)

    # Required for compute_qmedia
    dummy_data._n_tot_raw = n_tot - 365
    dummy_data.doy_climatology = np.zeros(366)

    dummy_data.segments = [(0, n_tot - 1)]

    # Needs a runmode for the fallback optimizer override
    dummy_data.runmode = 'PSO'
    dummy_data.time_res = "1d"
    dummy_data.n_particles = 5
    dummy_data.mod_num = "RK4"

    config = CVConfig(
        unit="year",
        n_years_per_fold=1,
        water_year_start_month=1,
        min_train_years=1,
        skip_first_year=True,
        min_valid_obs=1,
        optimizer_overrides={"n_run": 2, "n_particles": 2} # speed up test
    )

    results = run_leave_one_year_out_cv(dummy_data, config, 'PSO')

    assert len(results) > 0, "CV did not return any results"
    for r in results:
        assert r.nse >= -1e6  # Just verify it computed something

    df = summarize(results)
    assert not df.empty

    # Assert data was restored
    assert (dummy_data.Tair != -999.0).all()
    assert (dummy_data.Q != -999.0).all()

"""

if "def test_run_leave_one_year_out_cv_gap_tolerant" not in content:
    with open(filepath, 'a') as f:
        f.write(new_test)
    print("Successfully added test_run_leave_one_year_out_cv_gap_tolerant")
else:
    print("Test already exists")
