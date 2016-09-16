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

    def __init__(self, x_data, y_data, x_unit, y_area_unit="", is_spec_density=False, is_photon_flux=False):
        """
        Constructor of the spectrum y(x)

        :param is_spec_density:
        :param x_data: x data of the spectrum (1d numpy array)
        :param y_data: y data of the spectrum (1d numpy array)
        :param x_unit: the unit of x (string), e.g. 'nm', 'eV'
        :param y_area_unit: (string) If y is per area, put area unit here, e.g. 'm-2' or 'cm-2'.
        :param is_photon_flux: (boolean). True if y is number of photons.
        """

        self.set_spectrum(x_data=x_data, y_data=y_data, x_unit=x_unit,
                          y_area_unit=y_area_unit,
                          is_photon_flux=is_photon_flux,
                          is_spec_density=is_spec_density)

    def set_spectrum(self, x_data, y_data, x_unit, y_area_unit, is_photon_flux, is_spec_density):
        """


        """


        assert isinstance(x_data, np.ndarray)
        assert isinstance(y_data, np.ndarray)
        assert isinstance(x_unit, str)
        assert isinstance(y_area_unit, str)
        assert isinstance(is_photon_flux, bool)

        # Convert everything to photon energy : w/m^2-m

        self.is_spec_density = is_spec_density

        if y_area_unit != '':
            self.core_x, self.core_y = self.convert_spectrum_unit(x_data, y_data, from_x_unit=x_unit, to_x_unit='m',
                                                                  from_y_area_unit=y_area_unit, to_y_area_unit='m-2',
                                                                  is_spec_density=self.is_spec_density)
            self.y_area_unit = 'm-2'
        else:
            self.core_x, self.core_y = self.convert_spectrum_unit(x_data, y_data, from_x_unit=x_unit, to_x_unit='m',
                                                                  from_y_area_unit='', to_y_area_unit='',
                                                                  is_spec_density=self.is_spec_density)
            self.y_area_unit = ''

        # Convert photon flux to energy (J) representation
        if is_photon_flux:
            self.core_y = self._as_energy(self.core_x, self.core_y)

    def convert_spectrum_unit(self, x_data, y_data, from_x_unit, to_x_unit,
                              from_y_area_unit, to_y_area_unit,
                              is_spec_density):

        assert isinstance(x_data, np.ndarray)
        assert isinstance(y_data, np.ndarray)

        if x_data.size != y_data.size:
            raise ValueError("The array size of x_data and y_data do not match.")

        if us.compare_wavelength_dimension(from_x_unit, to_x_unit) == False:
            raise ValueError("The dimension of from_x_unit and to_x_unit do not match.")

        if us.compare_dimension(from_y_area_unit, to_y_area_unit) == False:
            raise ValueError("The dimension of from y_area_unit and to_y_area_unit do not match.")

        if is_spec_density:
            from_dx_unit = from_x_unit + "-1"
            to_dx_unit = to_x_unit + "-1"
        else:
            from_dx_unit = ""
            to_dx_unit = ""

        orig_y_div_unit = from_y_area_unit + " " + from_dx_unit
        new_orig_y_div_unit = to_y_area_unit + " " + to_dx_unit

        # Simple case
        if us.guess_dimension(from_x_unit) == us.guess_dimension(to_x_unit):

            new_x_data = us.convert(x_data, from_x_unit, to_x_unit)

            new_y_data = us.convert(y_data, orig_y_div_unit,
                                    new_orig_y_div_unit)


        elif us.guess_dimension(from_x_unit) == 'length' and us.guess_dimension(to_x_unit) == 'energy':

            new_x_data = us.convert(x_data, from_x_unit, 'nm')
            new_x_data = us.eVnm(new_x_data)
            new_x_data = us.convert(new_x_data, 'eV', to_x_unit)

            new_y_data = us.convert(y_data, from_y_area_unit, to_y_area_unit)

            if is_spec_density:
                conversion_constant = us.asUnit(sc.h, to_x_unit + " s") * us.asUnit(sc.c, from_x_unit + " s-1")
                new_y_data = new_y_data * conversion_constant / new_x_data ** 2


        elif us.guess_dimension(from_x_unit) == 'energy' and us.guess_dimension(to_x_unit) == 'length':

            new_x_data = us.convert(x_data, from_x_unit, 'eV')
            new_x_data = us.eVnm(new_x_data)
            new_x_data = us.convert(new_x_data, 'nm', to_x_unit)

            new_y_data = us.convert(y_data, from_y_area_unit, to_y_area_unit)

            if is_spec_density:
                conversion_constant = us.asUnit(sc.h, from_x_unit + " s") * us.asUnit(sc.c, to_x_unit + " s-1")
                new_y_data = new_y_data * conversion_constant / new_x_data ** 2

        return new_x_data, new_y_data

    def _x_set_spectrum(self, x_data, y_data, x_unit, is_photon_flux=False):

        """
        TO BE deprecated...........
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
            self.core_x = self._conv_wl_to_si(x_data, x_unit)

        elif x_unit in energy_wavelength_unit_factor.keys():
            self.core_x = self._conv_wl_to_si(x_data, x_unit)

        if is_photon_flux:
            y_data = self._as_energy(self.core_x, y_data)

        self.core_y = y_data

    def get_spectrum(self, x_unit, y_area_unit="default", flux="energy"):

        if y_area_unit == "default":
            y_area_unit = self.y_area_unit

        x_data, y_data = self.convert_spectrum_unit(self.core_x, self.core_y,
                                                    from_x_unit='m', to_x_unit=x_unit,
                                                    from_y_area_unit=self.y_area_unit, to_y_area_unit=y_area_unit,
                                                    is_spec_density=self.is_spec_density)

        # convert the spectrum to photon flux if necessary
        if flux == "photon":
            y_data = self._as_photon_flux(self.core_x, y_data)

        # Sort the spectrum by wavelength
        sorted_idx = np.argsort(x_data)
        x_data = x_data[sorted_idx]
        y_data = y_data[sorted_idx]

        return np.vstack((x_data, y_data))

    def old_get_spectrum_density(self, area_unit, wavelength_unit, flux="energy"):

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
            wl = us.asUnit(self.core_x, wavelength_unit)
            spec = us.convert(self.core_y, 'm-2', area_unit)
            spec = us.convert(spec, 'm-1', wavelength_unit + '-1')

        elif wavelength_unit in energy_wavelength_unit_factor.keys():
            wl = sc.h * sc.c / self.core_x
            wl = us.convert(wl, 'J', wavelength_unit)

            spec = us.convert(self.core_y, 'm-2', area_unit)
            spec = spec * np.power(self.core_x, 2) / (sc.h * sc.c)
            spec = us.convert(spec, 'J-1', wavelength_unit + '-1')

        # convert the spectrum to photon flux if necessary
        if flux == "photon":
            spec = self._as_photon_flux(self.core_x, spec)

        # Sort the spectrum by wavelength
        sorted_idx = np.argsort(wl)
        wl = wl[sorted_idx]
        spec = spec[sorted_idx]

        return np.vstack((wl, spec))

    def old_get_spectrum(self, x_unit, flux="energy"):

        length_wavelength_unit_factor = ('m', 'cm', 'nm')
        energy_wavelength_unit_factor = {"J": 1, "eV": sc.e}

        spectrum = np.zeros((2, self.core_x.shape[0]))

        if x_unit in length_wavelength_unit_factor:
            spectrum[0, :] = us.asUnit(self.core_x, x_unit)

        elif x_unit in energy_wavelength_unit_factor.keys():
            spectrum[0, :] = sc.h * sc.c / self.core_x
            spectrum[0, :] = us.convert(spectrum[0, :], 'J', x_unit)

        spectrum[1, :] = self.core_y

        # convert the spectrum to photon flux if necessary
        if flux == "photon":
            spectrum[1, :] = self._as_photon_flux(self.core_x, spectrum[1, :])

        # sort the spectrum by wavelength
        sorted_idx = np.argsort(spectrum[0, :])
        spectrum = spectrum[:, sorted_idx]

        return spectrum

    def get_interp_spectrum(self, x_data, x_unit, y_area_unit="default", flux="energy"):

        # TODO add code to check if the bound of x_data and self._core_wl overlaps
        # TODO merge get_interp_spectrum and get_interp_spectrum_density

        orig_spectrum = self.get_spectrum(x_unit, y_area_unit, flux)

        output_spectrum = np.zeros((2, x_data.shape[0]))

        output_spectrum[0, :] = x_data
        output_spectrum[1, :] = np.interp(x_data, orig_spectrum[0, :], orig_spectrum[1, :])

        return output_spectrum

    def old_get_interp_spectrum_density(self, x_data, y_area_unit, x_unit, flux="energy"):

        orig_spectral_density = self.get_spectrum_density(y_area_unit, x_unit, flux=flux)

        output_spectrum = np.zeros((2, x_data.shape[0]))

        output_spectrum[0, :] = x_data
        output_spectrum[1, :] = np.interp(x_data,
                                          orig_spectral_density[0, :], orig_spectral_density[1, :])

        return output_spectrum

    def __mul__(self, other):

        if type(other) == int or type(other) == float:
            newobj = copy.deepcopy(self)
            newobj.core_y = self.core_y * other
            return newobj
        elif isinstance(other, Spectrum):
            return self.attenuation_single(other, inplace=False)
        else:
            raise ValueError("The multipler should either be a scalar or a Spectrum calss object.")

    def attenuation_single(self, filter, inplace=True):
        assert isinstance(filter, Spectrum)

        atten_spec = filter.get_interp_spectrum(self.core_x, 'm')

        new_core_spec = self.core_y * atten_spec[1, :]

        if inplace:
            self.core_y = new_core_spec
            return None
        else:
            newobj = copy.deepcopy(self)
            newobj.core_y = new_core_spec
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

    def _conv_x_to_unit(self, x_data, from_x_unit, to_x_unit):

        if us.guess_dimension(from_x_unit) == us.guess_dimension(to_x_unit):
            return us.convert(x_data, from_unit=from_x_unit, to_unit=to_x_unit)

        elif (us.guess_dimension(from_x_unit) == 'energy' and us.guess_dimension(to_x_unit) == 'length'):

            new_x_data = us.convert(x_data, from_x_unit, 'J')
            new_x_data = us.mJ(new_x_data)

            return us.convert(new_x_data, 'm', to_x_unit)

        elif (us.guess_dimension(from_x_unit) == 'length' and us.guess_dimension(to_x_unit) == 'energy'):

            new_x_data = us.convert(x_data, from_x_unit, 'm')
            new_x_data = us.mJ(new_x_data)

            return us.convert(new_x_data, 'J', to_x_unit)

        else:
            raise ValueError("The input units (from_x_unit and to_x_unit) do not match")

    def _check_x_unit(self, from_x_unit, to_x_unit):

        d1 = us.guess_dimension(from_x_unit)
        d2 = us.guess_dimension(to_x_unit)

        if d1 == d2 and (d1 in ['length', 'energy']):
            return True
        elif set([d1, d2]) == set(['length', 'energy']):
            return True
        else:
            return False



if __name__ == "__main__":
    from illumination import illumination

    ill = illumination()

    test_spec = Spectrum(us.asUnit(ill.wl, 'nm'), ill.photon_flux_in_W / 1e9, "nm")

    # test_spec.set_spectrum_density(ill.wl_in_eV, ill.photon_flux_in_W_perEV, "m-2", "eV")

    import matplotlib.pyplot as plt

    # plt.plot(asUnit(test_spec.core_x,'nm'),test_spec.core_y/1e9,hold=True)

    nspectrum = test_spec.get_spectrum_density('m-2', 'm')

    plt.plot(nspectrum[0, :], nspectrum[1, :], hold=True)

    plt.plot(ill.wl, ill.photon_flux_in_W)
    plt.show()
