import unittest

import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
from pypvcell.solarcell import HighPSQCell, DiodeSeriesConnect
from pypvcell.illumination import load_astm

import matplotlib.pyplot as plt


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

    def test_diode_solver(self):
        hp1 = HighPSQCell(1.42, cell_T=300)
        hp1.set_input_spectrum(load_astm())

        hp2 = HighPSQCell(1.42, cell_T=300)
        hp2.set_input_spectrum(load_astm())

        hp3 = HighPSQCell(1.42, cell_T=300)
        hp3.set_input_spectrum(load_astm())

        ref_hp = HighPSQCell(1.42, cell_T=300)
        ref_hp.set_input_spectrum(load_astm())

        d1 = DiodeSeriesConnect([hp1, hp2, hp3])

        test_v = 1.0
        j1 = d1.get_j_from_v(test_v)
        j2 = ref_hp.get_j_from_v(test_v / 3.0)
        # by circuit law, j1 should equal to j2

        self.assertTrue(np.isclose(j1, j2))

        # extend the examination range

        test_v = np.linspace(-1, 1, 10)
        j1 = d1.get_j_from_v(test_v)
        j2 = ref_hp.get_j_from_v(test_v / 3.0)

        self.assertTrue(np.allclose(j1, j2))

    def test_multi_junction_diode(self):
        hp1 = HighPSQCell(1.87, cell_T=300)
        hp1.set_input_spectrum(load_astm())

        hp2 = HighPSQCell(1.42, cell_T=300)
        hp2.set_input_spectrum(load_astm())

        hp3 = HighPSQCell(1.0, cell_T=300)
        hp3.set_input_spectrum(load_astm())

        d1 = DiodeSeriesConnect([hp1, hp2, hp3])

        test_v = 1.0
        j1 = d1.get_j_from_v(test_v)

        test_v = np.linspace(-2, 3.4, num=200)
        j1 = d1.get_j_from_v(test_v)

        plt.plot(test_v, j1)
        plt.show()




if __name__ == '__main__':
    unittest.main()
