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
                Put null string '' if y does not have area unit
        :param is_photon_flux: (boolean). True if y is number of photons.
        """

        self.set_spectrum(x_data=x_data, y_data=y_data, x_unit=x_unit,
                          y_area_unit=y_area_unit,
                          is_photon_flux=is_photon_flux,
                          is_spec_density=is_spec_density)

    def set_spectrum(self, x_data, y_data, x_unit, y_area_unit, is_photon_flux, is_spec_density):
        """
        This is essentially a constructor method that sets up the attributes of the object.
        It converts everything to standard MKS unit. x_data: 'm', y_data: '[]/m^2-m'

        :param x_data:
        :param y_data:
        :param x_unit:
        :param y_area_unit:
        :param is_photon_flux:
        :param is_spec_density:
        :return: None
        """

        assert isinstance(x_data, np.ndarray)
        assert isinstance(y_data, np.ndarray)
        assert isinstance(x_unit, str)
        assert isinstance(y_area_unit, str)
        assert isinstance(is_photon_flux, bool)
        assert isinstance(is_spec_density, bool)

        # Convert everything to photon energy : [arb]/m^2-m

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

        """
        A general method for converting the spectrum y(x) from one set of unit to another.

        :param x_data: data of x (numpy array)
        :param y_data: data of y (numpy array)
        :param from_x_unit: the unit of x_data
        :param to_x_unit: the unit of x_data to be converted to
        :param from_y_area_unit: the unit of area of y_data
        :param to_y_area_unit: the unit of area of y_data tobe converted to
        :param is_spec_density: True if the data is spectral density.
        :return: a tuple of 1-d array (new_x_data, new_y_data)
        """

        assert isinstance(x_data, np.ndarray)
        assert isinstance(y_data, np.ndarray)
        assert isinstance(from_x_unit, str)
        assert isinstance(to_x_unit, str)
        assert isinstance(from_y_area_unit, str)
        assert isinstance(to_y_area_unit, str)
        assert isinstance(is_spec_density, bool)

        if x_data.size != y_data.size:
            raise ValueError("The array size of x_data and y_data do not match.")

        if not us.compare_wavelength_dimension(from_x_unit, to_x_unit):
            raise ValueError("The dimension of from_x_unit and to_x_unit do not match.")

        if not us.compare_dimension(from_y_area_unit, to_y_area_unit):
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

    def get_spectrum(self, to_x_unit, to_y_area_unit=None, to_photon_flux=False):
        """
        Retrieve the values of the spectrum based on the given units of x and y.

        :param to_x_unit: the unit of x
        :param to_y_area_unit: the unit of area of y. Default is the y_area_unit of the object.
        :param to_photon_flux: True if converting y to photon flux.
        :return: a 2xL numpy array
        """

        if to_y_area_unit is None:
            to_y_area_unit = self.y_area_unit

        x_data, y_data = self.convert_spectrum_unit(self.core_x, self.core_y,
                                                    from_x_unit='m', to_x_unit=to_x_unit,
                                                    from_y_area_unit=self.y_area_unit, to_y_area_unit=to_y_area_unit,
                                                    is_spec_density=self.is_spec_density)

        # convert the spectrum to photon flux if necessary
        if to_photon_flux:
            y_data = self._as_photon_flux(self.core_x, y_data)

        # Sort the spectrum by wavelength
        sorted_idx = np.argsort(x_data)
        x_data = x_data[sorted_idx]
        y_data = y_data[sorted_idx]

        return np.vstack((x_data, y_data))

    def get_interp_spectrum(self, to_x_data, to_x_unit, to_y_area_unit=None, to_photon_flux=False,
                            interp_left=None, interp_right=None):

        """
        Get interpolated spectrum.

        :param to_x_data: (ndarray) x values to be interpolated
        :param to_x_unit: (string) unit of x of the output value
        :param to_y_area_unit: (string) unit of area of y of the output value.
        :param to_photon_flux: (bool) True if converting the value to photon flux as the output
        :param interp_left: value to return for x < core_x[0], return core_x[0] if set to be None
        :param interp_right: value to return for x> core_x[-1], return core_y[-1] if set to be None
        :return: a 2xL array
        """

        orig_spectrum = self.get_spectrum(to_x_unit, to_y_area_unit, to_photon_flux=to_photon_flux)

        output_spectrum = np.zeros((2, to_x_data.shape[0]))

        output_spectrum[0, :] = to_x_data
        output_spectrum[1, :] = np.interp(to_x_data, orig_spectrum[0, :],
                                          orig_spectrum[1, :], left=interp_left, right=interp_right)

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
