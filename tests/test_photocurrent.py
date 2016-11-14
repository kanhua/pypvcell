__author__ = 'kanhua'

import unittest
from pypvcell.photocurrent import conv_abs_to_qe, calc_jsc, gen_step_qe, calc_jsc_from_eg
from pypvcell.illumination import Illumination
from pypvcell.spectrum import Spectrum
import numpy as np
import matplotlib.pyplot as plt


class MyTestCase(unittest.TestCase):

    def test_gen_qe_from_abs(self):

        abs_array=np.array([[300,1e7],[400,1e8]])
        layer_thickness=500e-6

        abs = Spectrum(abs_array[:, 0], abs_array[:, 1], x_unit='nm')

        qe=conv_abs_to_qe(abs,layer_thickness)

        qe_a = qe.get_spectrum(to_x_unit='nm')

        assert np.isclose(qe_a[1, 0], 1 - np.exp(-layer_thickness * abs_array[0, 1]))

    def test_gen_qe_from_abs_2(self):

        abs_file='./si_alpha.csv'
        abs_array=np.loadtxt(abs_file,delimiter=',')

        abs = Spectrum(abs_array[:, 0], abs_array[:, 1], x_unit='m')
        qe_1=conv_abs_to_qe(abs,500e-6)

        qe_2=conv_abs_to_qe(abs,5e-6)

        qe_1_a = qe_1.get_spectrum(to_x_unit='nm')

        qe_2_b = qe_2.get_spectrum(to_x_unit='nm')

        plt.plot(qe_1_a[0, :], qe_1_a[1, :], hold=True)
        plt.plot(qe_2_b[0, :], qe_2_b[1, :])
        plt.savefig("./test.png")

    def test_calc_jsc(self):
        """
        Test if calc_jsc and calc_jsc_from_eg return close results
        :return:
        """

        ill = Illumination(x_data="AM1.5g")
        qe = gen_step_qe(1.42, 1)

        jsc = calc_jsc(ill, qe)

        jsc2 = calc_jsc_from_eg(ill, 1.42)

        print("Jsc calculated by QE: %s" % jsc)
        print("Jsc calculated setting Eg and integrate the spectrum: %s" % jsc2)

        assert np.isclose(jsc, jsc2, rtol=5.e-3)


if __name__ == '__main__':
    unittest.main()
