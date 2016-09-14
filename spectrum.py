import numpy as np
import scipy.constants as sc
from units_system import UnitsSystem
import copy

us = UnitsSystem()


class Spectrum(object):
    """
    This class handles the operation of the spectrum y(x), including unit conversion and multiplication.


    It can handle unit conversions of different types of spectrum, including:

    1. Standard spectrum. The unit of y is independent of x, e.g. quantum efficiency, absorption spectrum, etc.
    2. Sepctral density. The unit of y is per [x-unit].
    For example, the Black-body radiation spectrum is often in the unit of energy/nm/m^2
    3. Photon flux: y is number of photons. When converting y into energy (J), it has to be multiplied by its photon energy.

    """

    def __init__(self, x_data, y_data, x_unit, y_area_unit=None, is_photon_flux=False):
        """
        Constructor of the spectrum y(x)

        :param x_data: x data of the spectrum (1d numpy array)
        :param y_data: y data of the spectrum (1d numpy array)
        :param x_unit: the unit of x (string), e.g. 'nm', 'eV'
        :param y_area_unit: (string) If y is per area, put area unit here, e.g. 'm-2' or 'cm-2'.
        :param is_photon_flux: (boolean). True if y is number of photons.
        """
        self.core_wl = None
        self.core_spec = None
        self.area_unit = y_area_unit
        self.is_photon_flux = is_photon_flux

        if y_area_unit is None:
            self.set_spectrum(x_data=x_data, y_data=y_data, x_unit=x_unit, is_photon_flux=is_photon_flux)
        else:
            self.set_spectrum_density(x_data=x_data, y_data=y_data, y_area_unit=y_area_unit, x_unit=x_unit,
                                      is_photon_flux=is_photon_flux)

    def set_spectrum_density(self, x_data, y_data, y_area_unit, x_unit, is_photon_flux=False):

        """
        Convert the input spectrum density to standard units defined in this class:
        spectrum density: W/m^2-m
        wavelength: m

        :param x_data: an ndarray that stores the wavelength
        :param y_data: an ndarray that stores the spectral density
        :param y_area_unit: the unit of the area, ex. m-2, cm-2
        :param x_unit: the unit of wavelength, ex: nm, m, eV, J
        :param is_photon_flux: True/False, specify whether the input spectrum is photon flux or not
        """

        assert isinstance(x_data, np.ndarray)
        assert isinstance(y_data, np.ndarray)

        flux_unit_factor = ["photon_flux", "J"]
        energy_wavelength_unit_factor = {"J": 1, "eV": sc.e}

        # Convert everything to photon energy : w/m^2-m

        if us.guess_dimension(x_unit) == 'length':

            # Convert [flux]/[Area][Length] to [Flux]/[m^2][m]
            self.core_spec = us.convert(y_data, y_area_unit + " " + x_unit + "-1", "m-2 m-1")

            # Convert wavelength unit to [m]
            self.core_wl = self._conv_wl_to_si(x_data, x_unit)

        elif x_unit in energy_wavelength_unit_factor.keys():
            self.core_wl = self._conv_wl_to_si(x_data, x_unit)

            # Convert [flux]/[Area][Energy] to [flux]/[Area][J] first, and then convert [flux]/[Area][J] to []/[Area][m]
            self.core_spec = us.convert(y_data, x_unit + '-1', 'J-1') * sc.h * sc.c / np.power(self.core_wl,
                                                                                               2)

            # Convert [flux]/[Area][Energy] to [flux]/[m^2][J]
            self.core_spec = us.convert(self.core_spec, y_area_unit, "m-2")

        if is_photon_flux:
            self.core_spec = self._as_energy(self.core_wl, self.core_spec)

    def set_spectrum(self, x_data, y_data, x_unit, is_photon_flux=False):

        """
        The spectrum can be anything that is a function of wavelength, ex. absorption coefficient, qe, etc.
        This method only converts the wavelength. For coverting spectral density, use read_speactrum_density()
        :param x_data: an ndarray that stores the wavelength
        :param y_data: an ndarray that stores the spectral density
        :param x_unit: length units (m, nm, ...) or "J" or "eV"
        """
        length_wavelength_unit_factor = ('m', 'cm', 'nm', 'um')
        energy_wavelength_unit_factor = {"J": 1, "eV": sc.e}

        # Convert everything to photon energy : w/m^2-m

        if x_unit in length_wavelength_unit_factor:
            self.core_wl = self._conv_wl_to_si(x_data, x_unit)

        elif x_unit in energy_wavelength_unit_factor.keys():
            self.core_wl = self._conv_wl_to_si(x_data, x_unit)

        if is_photon_flux:
            y_data = self._as_energy(self.core_wl, y_data)

        self.core_spec = y_data

    def get_spectrum_density(self, area_unit, wavelength_unit, flux="energy"):

        """

        :param area_unit:
        :param wavelength_unit:
        :param flux:
        :return:
        """

        length_wavelength_unit_factor = ('m', 'cm', 'nm')
        energy_wavelength_unit_factor = {"J": 1, "eV": sc.e}

        wl = None
        spec = None

        if wavelength_unit in length_wavelength_unit_factor:
            wl = us.asUnit(self.core_wl, wavelength_unit)
            spec = us.convert(self.core_spec, 'm-2', area_unit)
            spec = us.convert(spec, 'm-1', wavelength_unit + '-1')

        elif wavelength_unit in energy_wavelength_unit_factor.keys():
            wl = sc.h * sc.c / self.core_wl
            wl = us.convert(wl, 'J', wavelength_unit)

            spec = us.convert(self.core_spec, 'm-2', area_unit)
            spec = spec * np.power(self.core_wl, 2) / (sc.h * sc.c)
            spec = us.convert(spec, 'J-1', wavelength_unit + '-1')

        # convert the spectrum to photon flux if necessary
        if flux == "photon":
            spec = self._as_photon_flux(self.core_wl, spec)

        # Sort the spectrum by wavelength
        sorted_idx = np.argsort(wl)
        wl = wl[sorted_idx]
        spec = spec[sorted_idx]

        return np.vstack((wl, spec))

    def get_spectrum(self, x_unit, flux="energy"):

        length_wavelength_unit_factor = ('m', 'cm', 'nm')
        energy_wavelength_unit_factor = {"J": 1, "eV": sc.e}

        spectrum = np.zeros((2, self.core_wl.shape[0]))

        if x_unit in length_wavelength_unit_factor:
            spectrum[0, :] = us.asUnit(self.core_wl, x_unit)

        elif x_unit in energy_wavelength_unit_factor.keys():
            spectrum[0, :] = sc.h * sc.c / self.core_wl
            spectrum[0, :] = us.convert(spectrum[0, :], 'J', x_unit)

        spectrum[1, :] = self.core_spec

        # convert the spectrum to photon flux if necessary
        if flux == "photon":
            spectrum[1, :] = self._as_photon_flux(self.core_wl, spectrum[1, :])

        # sort the spectrum by wavelength
        sorted_idx = np.argsort(spectrum[0, :])
        spectrum = spectrum[:, sorted_idx]

        return spectrum

    def get_interp_spectrum(self, x_data, x_unit, flux="energy"):

        orig_spectrum = self.get_spectrum(x_unit, flux)

        output_spectrum = np.zeros((2, x_data.shape[0]))

        output_spectrum[0, :] = x_data
        output_spectrum[1, :] = np.interp(x_data, orig_spectrum[0, :], orig_spectrum[1, :])

        return output_spectrum

    def get_interp_spectrum_density(self, x_data, y_area_unit, x_unit, flux="energy"):

        orig_spectral_density = self.get_spectrum_density(y_area_unit, x_unit, flux=flux)

        output_spectrum = np.zeros((2, x_data.shape[0]))

        output_spectrum[0, :] = x_data
        output_spectrum[1, :] = np.interp(x_data,
                                          orig_spectral_density[0, :], orig_spectral_density[1, :])

        return output_spectrum

    def __mul__(self, other):

        if type(other) == int or type(other) == float:
            newobj = copy.deepcopy(self)
            newobj.core_spec = self.core_spec * other
            return newobj
        elif isinstance(other, Spectrum):
            return self.attenuation_single(other, inplace=False)
        else:
            raise ValueError("The multipler should either be a scalar or a Spectrum calss object.")

    def attenuation_single(self, filter, inplace=True):
        assert isinstance(filter, Spectrum)

        atten_spec = filter.get_interp_spectrum(self.core_wl, 'm')

        new_core_spec = self.core_spec * atten_spec[1, :]

        if inplace:
            self.core_spec = new_core_spec
            return None
        else:
            newobj = copy.deepcopy(self)
            newobj.core_spec = new_core_spec
            return newobj

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
        """
        Convert wavelength to meters.
        :param wavelength:
        :param unit:
        :return:
        """

        assert isinstance(unit, str)

        if unit == 'J':
            new_wl = us.mJ(wavelength)

        elif unit == 'eV':
            new_wl = us.eVnm(wavelength)
            new_wl = us.siUnits(new_wl, 'nm')

        else:
            new_wl = us.siUnits(wavelength, unit)

        return new_wl


if __name__ == "__main__":
    from illumination import illumination

    ill = illumination()

    test_spec = Spectrum(us.asUnit(ill.wl, 'nm'), ill.photon_flux_in_W / 1e9, "nm", area_unit="m-2")

    # test_spec.set_spectrum_density(ill.wl_in_eV, ill.photon_flux_in_W_perEV, "m-2", "eV")

    import matplotlib.pyplot as plt

    # plt.plot(asUnit(test_spec.core_wl,'nm'),test_spec.core_spec/1e9,hold=True)

    nspectrum = test_spec.get_spectrum_density('m-2', 'm')

    plt.plot(nspectrum[0, :], nspectrum[1, :], hold=True)

    plt.plot(ill.wl, ill.photon_flux_in_W)
    plt.show()
