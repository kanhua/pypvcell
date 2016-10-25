import numpy as np
import os
from scipy.interpolate import interp1d
from pypvcell.units_system import UnitsSystem
from pypvcell.spectrum import Spectrum
import pickle

us = UnitsSystem()


def load_default_spectrum(fname):
    cache_spectrum = {}
    spectrumfile = np.loadtxt(os.path.join(this_dir, "astmg173.csv"),
                              dtype=float, delimiter=',', skiprows=2)

    wl = spectrumfile[:, 0]

    cache_spectrum["wl"] = spectrumfile[:, 0]
    cache_spectrum["AM1.5g"] = spectrumfile[:, 2]
    cache_spectrum["AM1.5d"] = spectrumfile[:, 3]
    cache_spectrum["AM0"] = spectrumfile[:, 1]

    return cache_spectrum


# Read default spectrum
this_dir = os.path.split(__file__)[0]
spec_data = load_default_spectrum(os.path.join(this_dir, "astmg173.csv"))


class Illumination(Spectrum):
    def __init__(self, spectrum="AM1.5g", concentration=1):
        """
        Initialise a standard spectrum.
        """

        # flux, wl = self.read_from_csv(spectrum)

        wl = spec_data["wl"]
        flux = spec_data[spectrum]

        Spectrum.__init__(self, wl, flux * concentration, 'nm',
                          y_area_unit='m**-2', is_photon_flux=False, is_spec_density=True)

    def total_power(self):
        # Calculate power using different methods
        return np.trapz(self.core_y, self.core_x)


class BpFilter(Spectrum):
    def __init__(self, edge_in_eV, f_type="high_pass", OD=2, energy_bound=(0.5, 6)):

        """
        Create a band pass filter
        :param edge_in_eV: the cutoff frequency (in eV) of this filter
        :param f_type: high_pass or low_pass. high_pass: photons with energy higher than band edge passes.
        :param OD: optical density for attenuation
        :param energy_bound: the bound of wavelengths
        """
        a1 = np.linspace(energy_bound[0], edge_in_eV, num=100, endpoint=True)
        a2 = np.linspace(edge_in_eV + 0.01, energy_bound[1], num=100, endpoint=False)

        wavelength = np.concatenate((a1, a2))

        attenuation = np.zeros(wavelength.shape)

        if f_type == "high_pass":
            attenuation[wavelength <= edge_in_eV] = OD

        if f_type == "low_pass":
            attenuation[wavelength >= edge_in_eV] = OD

        attenuation = np.power(10, -attenuation)

        Spectrum.__init__(self, wavelength, attenuation, 'eV')


class material_filter(Spectrum):
    def __init__(self, material_abs, thickness):
        assert isinstance(material_abs, Spectrum)

        abs_spec = material_abs.get_spectrum(to_x_unit='m')

        attenuation = abs_spec[1, :] * thickness
        attenuation = np.exp(-attenuation)

        Spectrum.__init__(self, abs_spec[0, :], attenuation, 'm')


class QeFilter(Spectrum):
    """
    QeFilter is essentially a Spectrum of QE.
    It calculates the transmission of QE. i.e., this spectrum gives 1-EQE(E)
    """

    def __init__(self, qe_wavelength, qe_in_ratio, x_unit):
        Spectrum.__init__(self, qe_wavelength, 1 - qe_in_ratio, x_unit=x_unit)


if __name__ == "__main__":
    pass
