import numpy as np
import os
from scipy.interpolate import interp1d
from units_system import UnitsSystem
#from spectrum_base import spectrum_base
from spectrum_base_update import Spectrum
import pickle


us = UnitsSystem()

this_dir = os.path.split(__file__)[0]

with open(os.path.join(this_dir,'spec_data.pickle'), 'rb') as f:
    spec_data=pickle.load(f)


class illumination(Spectrum):
    def __init__(self, spectrum="AM1.5g", concentration=1):

        """
        Initialise a standard spectrum.
        """


        #flux, wl = self.read_from_csv(spectrum)

        wl=spec_data["wl"]
        flux=spec_data[spectrum]

        self.set_spectrum_density(wl, flux, "m-2", "nm")
        self.core_spec = self.core_spec * concentration

    def read_from_csv(self, spectrum):
        if spectrum in ["AM1.5g", "AM1.5d", "AM0"]:
            this_dir = os.path.split(__file__)[0]
            spectrumfile = np.loadtxt(os.path.join(this_dir, "astmg173.csv"),
                                      dtype=float, delimiter=',', skiprows=2)

            wl = spectrumfile[:, 0]

            if spectrum == "AM1.5g":
                flux = spectrumfile[:, 2]
            elif spectrum == "AM0":
                flux = spectrumfile[:, 1]
            elif spectrum == "AM1.5d":
                flux = spectrumfile[:, 3]
        return flux, wl

    def total_power(self):

        # Calculate power using different methods
        return np.trapz(self.core_spec, self.core_wl)


    def write_pc1d_abs(self, fname):

        current_dir = os.path.dirname(os.path.abspath(__file__))

        pc1d_ill = np.loadtxt(os.path.join(current_dir, "AM15D_trim.SPC"))

        pc1d_ill_per_nm = self.get_interp_spectrum_density(pc1d_ill[:, 0], "m-2", "nm")[:, 1]

        # fill delta wl

        delta_wl = np.zeros((pc1d_ill.shape[0],))

        delta_wl[0] = pc1d_ill[1, 0] - pc1d_ill[0, 0]
        delta_wl[-1] = pc1d_ill[-1, 0] - pc1d_ill[-2, 0]
        for idx in range(1, pc1d_ill.shape[0] - 1):
            delta_wl[idx] = ((pc1d_ill[idx + 1, 0] - pc1d_ill[idx, 0]) + (pc1d_ill[idx, 0] - pc1d_ill[idx - 1, 0])) / 2

        pc1d_ill_intensity = pc1d_ill_per_nm * delta_wl

        new_pc1d_ill = np.zeros(pc1d_ill.shape)

        new_pc1d_ill[:, 0] = pc1d_ill[:, 0]
        new_pc1d_ill[:, 1] = pc1d_ill_intensity

        np.savetxt(fname, new_pc1d_ill, fmt="%.3f")


class bp_filter(Spectrum):
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

        self.set_spectrum(wavelength, attenuation, 'eV')


class material_filter(Spectrum):
    def __init__(self, material_abs, thickness):
        assert isinstance(material_abs, Spectrum)
        attenuation = material_abs.core_spec * thickness
        attenuation = np.power(10, -attenuation)

        self.set_spectrum(material_abs.core_wl, attenuation, 'm')


class qe_filter_old(Spectrum):
    def __init__(self, wavelength_grid, qe_wl_in_ev, qe_in_ratio):
        assert isinstance(qe_wl_in_ev, np.ndarray)
        assert isinstance(qe_in_ratio, np.ndarray)
        assert qe_wl_in_ev.shape == qe_in_ratio.shape

        print("Warning: this QE filter can only deal with photon flux, "
              "it will ends up with wrong results if you convert the photon flux back to energy")

        QEinterp_func = interp1d(x=qe_wl_in_ev, y=qe_in_ratio)

        interped_qe_array = QEinterp_func(wavelength_grid)

        self.wavelength = wavelength_grid
        self.filter_attenuation = 1 - interped_qe_array


class qe_filter(Spectrum):
    def __init__(self, qe_wavelength, qe_in_ratio, wavelength_unit):
        self.set_spectrum(qe_wavelength, 1 - qe_in_ratio, wavelength_unit=wavelength_unit)

if __name__=="__main__":

    cache_spectrum={}
    spectrumfile = np.loadtxt(os.path.join(this_dir, "astmg173.csv"),
                              dtype=float, delimiter=',', skiprows=2)

    wl = spectrumfile[:, 0]

    cache_spectrum["wl"]=spectrumfile[:,0]
    cache_spectrum["AM1.5g"]=spectrumfile[:,2]
    cache_spectrum["AM1.5d"]=spectrumfile[:,3]
    cache_spectrum["AM0"]=spectrumfile[:,1]


    with open(os.path.join(this_dir,'spec_data.pickle'), 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(cache_spectrum, f)

