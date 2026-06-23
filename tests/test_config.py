import unittest
import numpy as np
from pyair2stream.config import CommonData, N_PAR, PI, TTT
import math

class TestConfig(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(N_PAR, 8)
        self.assertEqual(PI, np.float64(math.pi))
        self.assertEqual(TTT, np.float64(1.0 / 365.0))

    def test_common_data_initialization(self):
        cd = CommonData()
        self.assertEqual(cd.n_Q, 0)
        self.assertEqual(cd.Qmedia, np.float64(0.0))
        self.assertEqual(type(cd.Qmedia), np.float64)

        # Test optional numpy arrays
        self.assertIsNone(cd.I_pos)
        self.assertIsNone(cd.flag_par)

        # Test string
        self.assertEqual(cd.folder, "")

if __name__ == '__main__':
    unittest.main()
