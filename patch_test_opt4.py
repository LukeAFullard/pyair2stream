import sys

with open('tests/test_optimization.py', 'r') as f:
    code = f.read()

# Replace the test to properly set up cv_config for dummy data
code = code.replace("""        self.data.cross_validation = CVConfig(unit="year", skip_first_year=False, min_train_years=0, min_valid_obs=1)""", """        self.data.cross_validation = CVConfig(unit="year", skip_first_year=True, min_train_years=0, min_valid_obs=1)
        # Mock dates to ensure we have two years in dummy data so skip_first_year=True leaves 1 fold
        self.data.date = np.zeros((375, 3), dtype=np.int32)
        # Spinup year
        self.data.date[0:365, 0] = -999
        # Year 1 (skipped by skip_first_year=True)
        self.data.date[365:370, 0] = 2020
        # Year 2 (used as fold)
        self.data.date[370:375, 0] = 2021
        self.data.date[:, 1] = 1 # Month 1
        self.data.date[:, 2] = 1 # Day 1
""")

with open('tests/test_optimization.py', 'w') as f:
    f.write(code)
