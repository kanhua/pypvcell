import numpy as np
import os
from pypvcell.spectrum import Spectrum


def load_default_spectrum(fname1,fname2):
    cache_spectrum = {}
    spectrumfile = np.loadtxt(fname1,
                              dtype=float, delimiter=',', skiprows=2)

    cache_spectrum["AM1.5g"] = spectrumfile[:, [0,2]]
    cache_spectrum["AM1.5d"] = spectrumfile[:, [0,3]]
    cache_spectrum["AM0"] = spectrumfile[:, [0,1]]

    spectrumfile = np.loadtxt(fname2,
                              dtype=float, delimiter='\t')

    spectrumfile[:,1]/=1000
    cache_spectrum["AM1.5do"]=spectrumfile
    return cache_spectrum


# Read default spectrum
this_dir = os.path.split(__file__)[0]
spec_data = load_default_spectrum(os.path.join(this_dir, "astmg173.csv"),
                                  os.path.join(this_dir,"am15d.dat"))

class Illumination(Spectrum):
    def __init__(self, spectrum="AM1.5g", concentration=1):
        """
        Initialise a standard spectrum.
        """

        # flux, wl = self.read_from_csv(spectrum)


        flux = spec_data[spectrum]

        Spectrum.__init__(self, flux[:,0], flux[:,1] * concentration, 'nm',
                          y_unit='m**-2', is_photon_flux=False, is_spec_density=True)

    def total_power(self):
        # Calculate power using different methods
        return self.rsum()


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



if __name__ == "__main__":
    pass
