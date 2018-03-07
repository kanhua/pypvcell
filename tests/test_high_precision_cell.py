import unittest

import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
from pypvcell.solarcell import HighPSQCell, DiodeSeriesConnect
from pypvcell.illumination import load_astm


class TestHighPrecisionSQCellCase(unittest.TestCase):

    def setUp(self):
        self.dark_hpc = HighPSQCell(eg=1.42, cell_T=300)
        self.light_hpc = HighPSQCell(eg=1.42, cell_T=300)
        self.light_hpc.set_input_spectrum(load_astm())

    def test_reciprocal_equivalence_dark(self):
        """
        Calculate J0<-J(V0) and then V1<-V(J0)
        Then test if V0==V1

        :return:
        """
        test_v = np.linspace(-1, 1, 10)
        test_i = self.dark_hpc.get_j_from_v(test_v, to_tup=True)
        ttest_v = self.dark_hpc.get_v_from_j(test_i)

        self.assertTrue(np.allclose(test_v, ttest_v))

    def test_reciprocal_equivalence_light(self):
        """
        Calculate J0<-J(V0) and then V1<-V(J0)
        Then test if V0==V1
        This time with Jsc

        :return:
        """

        test_v = np.linspace(-1, 1, 10)
        test_i = self.light_hpc.get_j_from_v(test_v, to_tup=True)
        ttest_v = self.light_hpc.get_v_from_j(test_i)

        self.assertTrue(np.allclose(test_v, ttest_v))


if __name__ == '__main__':
    unittest.main()
