__author__ = 'kanhua'

import unittest
import numpy as np
from pypvcell.spectrum import Spectrum
import scipy.constants as sc
from pypvcell.units_system import UnitsSystem
from pypvcell.photocurrent import gen_step_qe
from pypvcell.illumination import Illumination
import matplotlib.pyplot as plt

us = UnitsSystem()


class SpectrumTestCases(unittest.TestCase):
    def setUp(self):
        # set up cases for length wavelength conversions
        self.init_wl = np.linspace(300, 1000, num=5)
        self.init_spec = np.ones((self.init_wl.shape[0],))
        self.spec_base = Spectrum(self.init_wl, self.init_spec, x_unit="nm", y_area_unit="m-2")

        # set up cases
        self.init_wl2 = np.linspace(1, 5, num=1000)
        self.init_spec2 = np.linspace(1, 5, num=1000)
        self.spec_base2 = Spectrum(self.init_wl2, self.init_spec2, x_unit="eV", y_area_unit="m-2", is_spec_density=True)

    def test_convert_spectrum_unit(self):
        x_data = np.linspace(300, 1000, num=3)
        y_data = np.ones(x_data.shape)

        # test without area units

        spec = Spectrum(x_data=x_data, y_data=y_data, x_unit='nm',
                        y_area_unit='', is_spec_density=False, is_photon_flux=False)

        new_x_data, new_y_data = spec.convert_spectrum_unit(x_data, y_data, from_x_unit='nm', to_x_unit='nm',
                                                            from_y_area_unit='', to_y_area_unit='',
                                                            is_spec_density=False)

        self.assertTrue(np.all(np.isclose(x_data, new_x_data)))
        assert np.all(np.isclose(y_data, new_y_data))

        # test with area units

        new_x_data, new_y_data = spec.convert_spectrum_unit(x_data, y_data, from_x_unit='nm', to_x_unit='nm',
                                                            from_y_area_unit='m-2', to_y_area_unit='cm-2',
                                                            is_spec_density=False)

        self.assertTrue(np.all(np.isclose(x_data, new_x_data)))
        self.assertTrue(np.all(np.isclose(y_data, new_y_data * 10000)))

        new_x_data, new_y_data = spec.convert_spectrum_unit(x_data, y_data, from_x_unit='nm', to_x_unit='eV',
                                                            from_y_area_unit='', to_y_area_unit='',
                                                            is_spec_density=False)

        self.assertTrue(np.all(np.isclose(y_data, new_y_data)))
        self.assertTrue(np.all(np.isclose(x_data, us.eVnm(new_x_data))))

        x_data = np.linspace(300, 1000, num=1000)  # use trapz to check the result, therefore num has to be large
        y_data = np.ones(x_data.shape)

        print("Test converting nm to eV")
        new_x_data, new_y_data = spec.convert_spectrum_unit(x_data, y_data, from_x_unit='nm', to_x_unit='eV',
                                                            from_y_area_unit='', to_y_area_unit='',
                                                            is_spec_density=True)

        self.assertTrue(np.all(np.isclose(x_data, us.eVnm(new_x_data))))
        area_after_conv = np.trapz(new_y_data[::-1], new_x_data[::-1])
        area_before_conv = np.trapz(y_data, x_data)
        self.assertTrue(np.isclose(area_before_conv, area_after_conv, rtol=1e-3))

        print("Test converting nm to eV with area:")
        new_x_data, new_y_data = spec.convert_spectrum_unit(x_data, y_data, from_x_unit='nm', to_x_unit='eV',
                                                            from_y_area_unit='m-2', to_y_area_unit='cm-2',
                                                            is_spec_density=True)

        self.assertTrue(np.all(np.isclose(x_data, us.eVnm(new_x_data))))
        area_after_conv = np.trapz(new_y_data[::-1], new_x_data[::-1])
        area_before_conv = np.trapz(y_data, x_data)
        self.assertTrue(np.isclose(area_before_conv, area_after_conv * 10000, rtol=1e-3))

        new_x_data, new_y_data = spec.convert_spectrum_unit(x_data, y_data, from_x_unit='nm', to_x_unit='eV',
                                                            from_y_area_unit='m-2', to_y_area_unit='m-2',
                                                            is_spec_density=True)

        self.assertTrue(np.all(np.isclose(x_data, us.eVnm(new_x_data))))
        area_after_conv = np.trapz(new_y_data[::-1], new_x_data[::-1])
        area_before_conv = np.trapz(y_data, x_data)
        self.assertTrue(np.isclose(area_before_conv, area_after_conv, rtol=1e-3))

        print("Test converting eV to nm")
        x_data = np.linspace(0.2, 5, num=1000)  # use trapz to check the result, therefore num has to be large
        y_data = np.ones(x_data.shape)

        new_x_data, new_y_data = spec.convert_spectrum_unit(x_data, y_data, from_x_unit='eV', to_x_unit='nm',
                                                            from_y_area_unit='', to_y_area_unit='',
                                                            is_spec_density=True)

        self.assertTrue(np.all(np.isclose(x_data, us.eVnm(new_x_data))))
        area_after_conv = np.trapz(new_y_data[::-1], new_x_data[::-1])
        area_before_conv = np.trapz(y_data, x_data)
        self.assertTrue(np.isclose(area_before_conv, area_after_conv, rtol=1e-3))

        x_data = np.linspace(0.2, 5, num=1000)  # use trapz to check the result, therefore num has to be large
        y_data = np.ones(x_data.shape)

        print("Test converting eV to nm, per area:")
        new_x_data, new_y_data = spec.convert_spectrum_unit(x_data, y_data, from_x_unit='eV', to_x_unit='nm',
                                                            from_y_area_unit='m-2', to_y_area_unit='cm-2',
                                                            is_spec_density=True)

        area_after_conv = np.trapz(new_y_data[::-1], new_x_data[::-1])
        area_before_conv = np.trapz(y_data, x_data)

        self.assertTrue(np.all(np.isclose(x_data, us.eVnm(new_x_data))))

        self.assertTrue(np.isclose(area_before_conv, area_after_conv * 10000, rtol=1e-2))

    def test_1(self):
        spectrum = self.spec_base.get_spectrum("nm", 'm-2')
        assert np.all(np.isclose(spectrum[0, :], self.init_wl))
        assert np.all(np.isclose(spectrum[1, :], self.init_spec))

        spectrum = self.spec_base.get_spectrum('nm', 'cm-2')
        assert np.all(np.isclose(spectrum[0, :], self.init_wl))
        assert np.all(np.isclose(spectrum[1, :], self.init_spec / 1e4))

        # This is not spectral density, therefore we don't have to convert nm->m in sefl.init_spec
        spectrum = self.spec_base.get_spectrum('m', 'cm-2')
        assert np.all(np.isclose(spectrum[0, :], self.init_wl / 1e9))
        assert np.all(np.isclose(spectrum[1, :], self.init_spec / 1e4))

    def test_4(self):
        init_wl2 = np.linspace(1, 5, num=1000)
        init_spec2 = np.linspace(1, 5, num=1000)

        spec_base2 = Spectrum(init_wl2, init_spec2, x_unit="eV", y_area_unit="", is_spec_density=True)

        spectrum = spec_base2.get_spectrum(to_x_unit='nm', to_y_area_unit='')

        self.assertTrue(np.all(np.isclose(spectrum[0, :], np.sort(us.eVnm(init_wl2)))))

        area_before = np.trapz(init_spec2, init_wl2)
        area_after = np.trapz(spectrum[1, :], spectrum[0, :])

        self.assertTrue(np.isclose(area_before, area_after),
                        msg="area_before: %s, area_after: %s" % (area_before, area_after))

    def test_5(self):
        spectrum = self.spec_base2.get_spectrum(to_x_unit='J', to_y_area_unit="m-2")

        assert np.all(np.isclose(spectrum[0, :], np.sort(self.init_wl2) * sc.e))
        assert np.isclose(np.trapz(spectrum[1, :], spectrum[0, :]), np.trapz(self.init_spec2, self.init_wl2))

    def test_6(self):
        init_wl = np.linspace(1, 5, num=10)
        init_spec = np.ones(init_wl.shape)

        test_spec_base = Spectrum(x_data=init_wl, y_data=init_spec, x_unit='eV', y_area_unit="")
        spectrum = test_spec_base.get_spectrum(to_x_unit='nm', to_y_area_unit="")

        assert np.all(np.isclose(spectrum[0, :], np.sort(us.eVnm(init_wl))))

    def test_energy_flux_conversion(self):
        """
        This test converts photon flux to energy flux

        """
        init_wl = np.linspace(300, 500, num=10)
        init_spec = np.ones(init_wl.shape)

        test_spec_base = Spectrum(init_wl, init_spec, x_unit='nm', is_photon_flux=True)
        spectrum = test_spec_base.get_spectrum(to_x_unit='nm')

        # Prepare an expected spectrum for comparsion
        expect_spec = init_spec * sc.h * sc.c / us.siUnits(init_wl, 'nm')

        # Since the values of the spectrum are very small, causing the errors in np.isclose()
        # ( both are in the order of ~1e-19) Need renormalise them for proper comparison.
        assert np.all(np.isclose(spectrum[1, :] * 1e19, expect_spec * 1e19))

    def test_photon_flux_conversion(self):
        """
        This test converts energy flux to photons

        """
        init_wl = np.linspace(300, 500, num=10)
        init_spec = np.ones(init_wl.shape)

        test_spec_base = Spectrum(init_wl, init_spec, 'nm', is_photon_flux=False)
        spectrum = test_spec_base.get_spectrum('nm', to_photon_flux=True)

        expect_spec = init_spec / (sc.h * sc.c / us.siUnits(init_wl, 'nm'))

        assert np.all(np.isclose(spectrum[1, :], expect_spec))


    def test_wrong_units(self):

        init_wl = np.linspace(300, 500, num=10)
        init_spec = np.ones(init_wl.shape)

        test_spec_base = Spectrum(init_wl, init_spec, 'nm', is_photon_flux=False)

        with self.assertRaises(ValueError):
            test_spec_base.get_spectrum(to_x_unit='s')

    def test_9(self):
        """
        This test sets up a spectrum, and filter it with GaAs substrate.
        Unfiltered part of the spectrum has 10% loss.
        This essentially cut the spectrum at 1.42 eV
        :return:
        """

        sq_qe = gen_step_qe(1.42, 0.9)
        test_ill = Illumination()
        # test_qef = qe_filter(sq_qe)

        filtered_ill = test_ill * sq_qe

        assert isinstance(filtered_ill, Illumination)

        plt.plot(filtered_ill.get_spectrum('eV')[0, :], filtered_ill.get_spectrum('eV')[1, :], label="filtered")
        plt.plot(test_ill.get_spectrum('eV')[0, :], test_ill.get_spectrum('eV')[1, :], label="original")

        plt.xlabel('wavelength (eV)')
        plt.ylabel('spectrum (W/eV/m^2)')

        plt.legend()

        plt.show()

    def test_mul_scalar(self):
        init_wl = np.linspace(300, 500, num=10)
        init_spec = np.ones(init_wl.shape)

        test_spec_base = Spectrum(init_wl, init_spec, 'nm', is_photon_flux=False)
        test_spec_base = test_spec_base * 0.5

        spectrum = test_spec_base.get_spectrum('nm')

        self.assertEqual(spectrum[1, 5], 0.5)

    def test_mul_spectrum(self):
        init_wl = np.linspace(300, 500, num=10)
        init_spec = np.ones(init_wl.shape)

        mulp_wl = np.linspace(200, 600, num=10)
        mulp_spec = np.ones(init_wl.shape) * 0.5

        test_spec_base = Spectrum(init_wl, init_spec, 'nm', is_photon_flux=False)
        mulp_spec_base = Spectrum(mulp_wl, mulp_spec, 'nm', is_photon_flux=False)
        test_spec_base = test_spec_base * mulp_spec_base

        spec_arr = test_spec_base.get_spectrum('nm')

        self.assertEqual(spec_arr[1, 5], 0.5)

    def test_mul_wrong_spectrum(self):
        init_wl = np.linspace(300, 500, num=10)
        init_spec = np.ones(init_wl.shape)
        test_spec_base = Spectrum(init_wl, init_spec, 'nm', is_photon_flux=False)

        with self.assertRaises(TypeError):
            test_spec_base * 'r'

    def test_inv_op(self):
        init_wl = np.linspace(300, 500, num=10)
        init_spec = np.ones(init_wl.shape)

        s1 = Spectrum(init_wl, init_spec, 'nm', is_photon_flux=False)

        s3 = 1 - s1 * 0.2
        s3_c = s1 * 0.8
        s3_c2 = 1 + s1 * (-0.2)

        self.assertTrue(np.allclose(s3.core_y, s3_c.core_y))

        self.assertTrue(np.allclose(s3.core_y, s3_c2.core_y))

        s4 = 1 / (s1 * 2)
        s4_c = s1 * 0.5

        self.assertTrue(np.allclose(s4.core_y, s4_c.core_y))


if __name__ == '__main__':
    unittest.main()
