import sys

with open('tests/test_optimization.py', 'r') as f:
    code = f.read()

# Replace the test to properly set up cv_config for dummy data
code = code.replace("""        self.data.uncertainty_options = {"noise_model": "iid", "ar1_rho": None}

        DE_CV_MCMC_mode(self.data, seed=42)""", """        self.data.uncertainty_options = {"noise_model": "iid", "ar1_rho": None}
        from pyair2stream.cross_validation import CVConfig
        self.data.cross_validation = CVConfig(unit="year", skip_first_year=False, min_train_years=0, min_valid_obs=1)

        DE_CV_MCMC_mode(self.data, seed=42)""")

with open('tests/test_optimization.py', 'w') as f:
    f.write(code)
