import unittest
import numpy as np

from iii_v_si import calc_gaas


class MyTestCase(unittest.TestCase):
    def test_gaas_on_si_2J(self):
        eta_max, topeqe = calc_gaas()

        self.assertTrue(np.isclose(eta_max, 0.35759),
                        msg="the maximum efficiency of GaAs/Si is wrongly calculated")
        self.assertTrue(np.isclose(topeqe, 0.68163),
                        msg="the optimal EQE of GaAs on silicon is wrongly calculated")


if __name__ == '__main__':
    unittest.main()
