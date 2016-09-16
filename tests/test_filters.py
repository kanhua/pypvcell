__author__ = 'kanhua'

import unittest
from illumination import bp_filter, qe_filter, illumination
import numpy as np
from photocurrent import gen_square_qe_array
import matplotlib.pyplot as plt
from units_system import UnitsSystem

us = UnitsSystem()


class FiltersTestCase(unittest.TestCase):
    def test_bpFilter(self):
        bf = bp_filter(x_data=1.42, y_data=2, x_unit=null)
        abs_spec = bf.get_spectrum(to_x_unit='nm')

        # Test whether the absorption in the range between 0.1 to 1 eV equals to 0.01
        test_wl = np.linspace(0.1, 1, num=10)
        interped_abs_spec = bf.get_interp_spectrum(test_wl, 'eV')
        expected_spec = np.ones((interped_abs_spec.shape[0],)) * 0.01
        assert np.all(np.isclose(interped_abs_spec[:, 1], expected_spec))

    def test_qe_filter(self):
        qe_below_edge = 1e-3
        qe_value = 0.9
        qe_array = gen_square_qe_array(1.42, qe_value, qe_below_edge=qe_below_edge)
        qe_abs = qe_filter(qe_array[:, 0], qe_array[:, 1], 'eV')

        test_wl = np.linspace(300, 500, num=10)
        output_abs_spectrum = qe_abs.get_interp_spectrum(test_wl, 'nm')

        test_wl_2 = np.linspace(1000, 1200, num=10)
        output_abs_spectrum2 = qe_abs.get_interp_spectrum(test_wl_2, 'nm')

        test_wl_3 = np.linspace(0.2, 1.41, num=10)
        output_abs_spectrum3 = qe_abs.get_interp_spectrum(test_wl_3, 'nm')

        assert np.all(np.isclose(output_abs_spectrum[:, 1], 1 - np.ones(test_wl.shape) * qe_value))

        assert np.all(np.isclose(output_abs_spectrum2[:, 1], 1 - np.ones(test_wl_2.shape) * qe_below_edge))

        assert np.all(np.isclose(output_abs_spectrum3[:, 1], 1 - np.ones(test_wl_3.shape) * qe_value))

    def test_draw_spectrum(self):
        ill = illumination()

        spec = ill.get_spectrum_density('m-2', 'nm')

        plt.plot(spec[:, 0], spec[:, 1])

        plt.savefig("test_ill_spectrum.png")

        # Check whether the range of the wavelengths are correct
        # Expect the illmination wavelength range between 100nm and 9000nm
        self.assertLess(np.max(spec[:, 0]), 9000)
        self.assertLess(100, np.min(spec[:, 0]))

    def test_draw_filtered_spectrum(self):
        edge = 1.42
        ill = illumination()
        orig_spec = ill.get_spectrum_density('m-2', 'nm')

        ill.attenuation_single(bp_filter(x_data=edge, y_data=2, x_unit=null))

        filtered_spec = ill.get_spectrum_density('m-2', 'nm')

        plt.plot(filtered_spec[:, 0], filtered_spec[:, 1])
        plt.savefig("filtered_spec.png")

        # Due to the discretization error, we need to add a buffer (10nm), not cutting at the edge
        above_bg_spec = orig_spec[orig_spec[:, 0] < (us.eVnm(edge) - 10), :]

        filtered_above_bg_spec = filtered_spec[orig_spec[:, 0] < (us.eVnm(edge) - 10), :]

        assert np.all(np.isclose(above_bg_spec[:, 1], filtered_above_bg_spec[:, 1]))


if __name__ == '__main__':
    unittest.main()
