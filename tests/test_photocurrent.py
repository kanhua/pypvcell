__author__ = 'kanhua'

import unittest
from photocurrent import gen_qe_from_abs
from spectrum_base import spectrum_base
import numpy as np
import matplotlib.pyplot as plt


class MyTestCase(unittest.TestCase):

    def test_gen_qe_from_abs(self):

        abs_array=np.array([[300,1e7],[400,1e8]])
        layer_thickness=500e-6

        abs=spectrum_base()
        abs.set_spectrum(abs_array[:,0],abs_array[:,1],wavelength_unit='nm')

        qe=gen_qe_from_abs(abs,layer_thickness)

        qe_a=qe.get_spectrum(wavelength_unit='nm')

        assert np.isclose(qe_a[0,1],1-np.exp(-layer_thickness*abs_array[0,1]))

    def test_gen_qe_from_abs_2(self):

        abs_file='./si_alpha.csv'
        abs_array=np.loadtxt(abs_file,delimiter=',')

        abs=spectrum_base()
        abs.set_spectrum(abs_array[:,0],abs_array[:,1],wavelength_unit='m')

        qe_1=gen_qe_from_abs(abs,500e-6)

        qe_2=gen_qe_from_abs(abs,5e-6)

        qe_1_a=qe_1.get_spectrum(wavelength_unit='nm')

        qe_2_b=qe_2.get_spectrum(wavelength_unit='nm')


        plt.plot(qe_1_a[:,0],qe_1_a[:,1],hold=True)
        plt.plot(qe_2_b[:,0],qe_2_b[:,1])
        plt.show()



if __name__ == '__main__':
    unittest.main()
