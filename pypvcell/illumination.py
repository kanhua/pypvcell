import numpy as np
import os
import scipy.constants as sc
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

def load_blackbody(T=6000,normalize_to=None):
    """
    Load Blackbody spectrum

    :param T: temperature
    :param normalize_to: the value in W/m^2 that the output spectrum is normalized to. Set to None if no renormalization is required.
    :return: Spectrum
    """

    # Initialze the wavelength in nm-> m
    wl=np.arange(20,2000,step=20)/1e9

    # Convert it to frequency
    mu=sc.c/wl

    # Intensity of Blackbody spectrum in (W/m^2)
    blackbody_i=2*sc.pi*sc.h*np.power(mu,3)/np.power(sc.c,2)*(1/(np.exp(sc.h*mu/sc.k/T)-1))

    factor=1
    sp=Spectrum(x_data=mu, y_data=blackbody_i, x_unit='s**-1',
             y_unit="m**-2", is_spec_density=True, is_photon_flux=False)

    if normalize_to is not None:
        factor=normalize_to/sp.rsum()

    return sp*factor


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
