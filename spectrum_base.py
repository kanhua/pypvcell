import numpy as np
import scipy.constants as sc
from units_system import UnitsSystem
import copy

us=UnitsSystem()

class spectrum_base(object):
    def __init__(self):
        self.core_wl = np.zeros((1,))
        self.core_spec = np.zeros((1,))
        pass

    def set_spectrum_density(self, wavelength, spectrum, area_unit, wavelength_unit, is_photon_flux=False):


        """
        Convert the input spectrum density to standard units defined in this class:
        spectrum density: W/m^2-m
        wavelength: m
        :param wavelength: an ndarray that stores the wavelength
        :param spectrum: an ndarray that stores the spectral density
        :param area_unit: the unit of the area, ex. m-2, cm-2
        :param wavelength_unit: the unit of wavelength, ex: nm, m, eV, J
        :param is_photon_flux: True/False, specify whether the input spectrum is photon flux or not
        """

        isinstance(wavelength, np.ndarray)
        isinstance(spectrum, np.ndarray)

        flux_unit_factor = ["photon_flux", "J"]
        length_wavelength_unit_factor = ('m', 'cm', 'nm')
        energy_wavelength_unit_factor = {"J": 1, "eV": sc.e}

        # Convert everything to photon energy : w/m^2-m

        if wavelength_unit in length_wavelength_unit_factor:

            # Convert [flux]/[Area][Length] to [Flux]/[m^2][m]
            self.core_spec = us.convert(spectrum, area_unit + " " + wavelength_unit + "-1", "m-2 m-1")

            # Convert wavelength unit to [m]
            self.core_wl = self._conv_wl_to_si(wavelength, wavelength_unit)

        elif wavelength_unit in energy_wavelength_unit_factor.keys():
            self.core_wl = self._conv_wl_to_si(wavelength, wavelength_unit)

            # Convert [flux]/[Area][Energy] to [flux]/[Area][J] first, and then convert [flux]/[Area][J] to []/[Area][m]
            self.core_spec = us.convert(spectrum, wavelength_unit + '-1', 'J-1') * sc.h * sc.c / np.power(self.core_wl, 2)

            # Convert [flux]/[Area][Energy] to [flux]/[m^2][J]
            self.core_spec = us.convert(self.core_spec, area_unit, "m-2")

        if is_photon_flux:
            self.core_spec = self._as_energy(self.core_wl, self.core_spec)

    def set_spectrum(self, wavelength, spectrum, wavelength_unit, is_photon_flux=False):


        """
        " Load the input spectrum into the class.
        The spectrum can be anything that is a function of wavelength, ex. absorption coefficient, qe, etc.
        This method only converts the wavelength. For coverting spectral density, use read_speactrum_density()
        :param wavelength: an ndarray that stores the wavelength
        :param spectrum: an ndarray that stores the spectral density
        :param wavelength_unit: length units (m, nm, ...) or "J" or "eV"
        """
        length_wavelength_unit_factor = ('m', 'cm', 'nm')
        energy_wavelength_unit_factor = {"J": 1, "eV": sc.e}

        # Convert everything to photon energy : w/m^2-m

        if wavelength_unit in length_wavelength_unit_factor:
            self.core_wl = self._conv_wl_to_si(wavelength, wavelength_unit)

        elif wavelength_unit in energy_wavelength_unit_factor.keys():
            self.core_wl = self._conv_wl_to_si(wavelength, wavelength_unit)

        if is_photon_flux:
            spectrum = self._as_energy(self.core_wl, spectrum)

        self.core_spec = spectrum


    def get_spectrum_density(self, area_unit, wavelength_unit, flux="energy"):

        length_wavelength_unit_factor = ('m', 'cm', 'nm')
        energy_wavelength_unit_factor = {"J": 1, "eV": sc.e}

        spectrum = np.zeros((self.core_wl.shape[0], 2))

        if wavelength_unit in length_wavelength_unit_factor:
            spectrum[:, 0] = us.asUnit(self.core_wl, wavelength_unit)
            spectrum[:, 1] = us.convert(self.core_spec, 'm-2', area_unit)
            spectrum[:, 1] = us.convert(spectrum[:, 1], 'm-1', wavelength_unit + '-1')

        elif wavelength_unit in energy_wavelength_unit_factor.keys():
            spectrum[:, 0] = sc.h * sc.c / self.core_wl
            spectrum[:, 0] = us.convert(spectrum[:, 0], 'J', wavelength_unit)

            spectrum[:, 1] = us.convert(self.core_spec, 'm-2', area_unit)
            spectrum[:, 1] = spectrum[:, 1] * np.power(self.core_wl, 2) / (sc.h * sc.c)
            spectrum[:, 1] = us.convert(spectrum[:, 1], 'J-1', wavelength_unit + '-1')

        # convert the spectrum to photon flux if necessary
        if flux == "photon":
            spectrum[:, 1] = self._as_photon_flux(self.core_wl, spectrum[:, 1])



        # Sort the spectrum by wavelength
        sorted_idx = np.argsort(spectrum[:, 0])
        spectrum = spectrum[sorted_idx, :]

        return spectrum

    def get_spectrum(self, wavelength_unit, flux="energy"):

        length_wavelength_unit_factor = ('m', 'cm', 'nm')
        energy_wavelength_unit_factor = {"J": 1, "eV": sc.e}

        spectrum = np.zeros((self.core_wl.shape[0], 2))

        if wavelength_unit in length_wavelength_unit_factor:
            spectrum[:, 0] = us.asUnit(self.core_wl, wavelength_unit)

        elif wavelength_unit in energy_wavelength_unit_factor.keys():
            spectrum[:, 0] = sc.h * sc.c / self.core_wl
            spectrum[:, 0] = us.convert(spectrum[:, 0], 'J', wavelength_unit)

        spectrum[:, 1] = self.core_spec

        # convert the spectrum to photon flux if necessary
        if flux == "photon":
            spectrum[:, 1] = self._as_photon_flux(self.core_wl, spectrum[:, 1])

        # sort the spectrum by wavelength
        sorted_idx = np.argsort(spectrum[:, 0])
        spectrum = spectrum[sorted_idx, :]

        return spectrum

    def get_interp_spectrum(self, wavelength, wavelength_unit, flux="energy"):

        orig_spectrum = self.get_spectrum(wavelength_unit, flux)

        output_spectrum = np.zeros((wavelength.shape[0], 2))

        output_spectrum[:, 0] = wavelength
        output_spectrum[:, 1] = np.interp(wavelength, orig_spectrum[:, 0], orig_spectrum[:, 1])

        return output_spectrum

    def get_interp_spectrum_density(self, wavelength, area_unit, wavelength_unit, flux="energy"):

        orig_spectral_density = self.get_spectrum_density(area_unit, wavelength_unit, flux=flux)

        output_spectrum = np.zeros((wavelength.shape[0], 2))

        output_spectrum[:, 0] = wavelength
        output_spectrum[:, 1] = np.interp(wavelength,
                                          orig_spectral_density[:, 0], orig_spectral_density[:, 1])

        return output_spectrum


    def attenuation_single(self, filter):

        assert isinstance(filter, spectrum_base)

        atten_spec = filter.get_interp_spectrum(self.core_wl, 'm')

        self.core_spec = self.core_spec * atten_spec[:, 1]

        return copy.deepcopy(self)

    def attenuation(self, filters):


        """
        Do attenuation of a list of filters
        :param filters: a list of filter instances
        """
        assert isinstance(filters, list)

        for f in filters:
            self.attenuation_single(f)

        return copy.deepcopy(self)


    def _as_photon_flux(self, wavelength, energy_flux):

        return energy_flux / (sc.h * sc.c) * wavelength


    def _as_energy(self, wavelength, photon_flux):

        return photon_flux * (sc.h * sc.c) / wavelength

    def _conv_wl_to_si(self, wavelength, unit):

        assert isinstance(unit, str)

        energy_unit = {"J": 1, 'eV': sc.e}

        if unit in energy_unit:
            # self.wl_in_eV = sc.h * sc.c / (self.wl * sc.e)
            new_wl = sc.h * sc.c / (wavelength * energy_unit[unit])
        else:
            new_wl = us.siUnits(wavelength, unit)

        return new_wl


if __name__ == "__main__":
    test_spec = spectrum_base()
    from illumination import illumination

    ill = illumination()

    test_spec.set_spectrum_density(us.asUnit(ill.wl, 'nm'), ill.photon_flux_in_W / 1e9, "m-2", "nm")

    test_spec.set_spectrum_density(ill.wl_in_eV, ill.photon_flux_in_W_perEV, "m-2", "eV")

    import matplotlib.pyplot as plt

    # plt.plot(asUnit(test_spec.core_wl,'nm'),test_spec.core_spec/1e9,hold=True)

    nspectrum = test_spec.get_spectrum_density('m-2', 'm')

    plt.plot(nspectrum[:, 0], nspectrum[:, 1], hold=True)

    plt.plot(ill.wl, ill.photon_flux_in_W)
    plt.show()