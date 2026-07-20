import sys

with open('tests/test_optimization.py', 'r') as f:
    code = f.read()

new_code = code + """
    def test_DE_CV_MCMC_mode(self):
        from pyair2stream.optimization import DE_CV_MCMC_mode
        self.data.n_particles = 2
        self.data.n_run = 1
        self.data.mcmc_walkers = 16
        self.data.mcmc_steps = 10
        self.data.runmode = 'DE-CV-MCMC'
        self.data.uncertainty_options = {"noise_model": "iid", "ar1_rho": None}

        DE_CV_MCMC_mode(self.data, seed=42)

        chain_path = os.path.join(self.data.folder, f"MCMC_chain_test_station_test_series_1d.csv")
        sidecar_path = os.path.join(self.data.folder, f"MCMC_chain_test_station_test_series_1d_meta.json")
        env_path = os.path.join(self.data.folder, f"MCMC_envelopes_test_station_test_series_1d.csv")

        self.assertTrue(os.path.exists(chain_path))
        self.assertTrue(os.path.exists(sidecar_path))
        self.assertTrue(os.path.exists(env_path))
"""

with open('tests/test_optimization.py', 'w') as f:
    f.write(new_code)
