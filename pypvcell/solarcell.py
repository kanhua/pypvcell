from pypvcell.illumination import Illumination
from pypvcell.photocurrent import gen_step_qe, calc_jsc_from_eg, calc_jsc
from pypvcell.ivsolver import calculate_j01, gen_rec_iv_by_rad_eta, solve_mj_iv
from pypvcell.fom import max_power
from pypvcell.spectrum import Spectrum
from pypvcell.detail_balanced_MJ import calculate_j01_from_qe
import numpy as np
import scipy.constants as sc

thermal_volt = sc.k / sc.e


class SolarCell(object):
    def get_iv(self):
        raise NotImplementedError()

    def set_input_spectrum(self, input_spectrum):
        """
        Set the illlumination spectrum of the solar cell

        :param input_spectrum: the illumination spectrum.
        :type input_spectrum: Illumination
        """
        raise NotImplementedError()

    def get_transmit_spectrum(self):
        """

        :return: The transmitted spectrum of this solar cell
        :rtype: Illumination
        """

        raise NotImplementedError()


class SQCell(SolarCell):
    """
    A SolarCell at Shockley-Queisser limit

    """

    def __init__(self, eg, cell_T, n_c=3.5, n_s=1):
        """
        Initialize a SQ solar cell.
        It loads the class and sets up J01 of the cell

        :param eg: Band gap (eV)
        :param cell_T: temperature (K)
        :param n_c: refractive index of cell
        :param n_s: refractive index of ambient
        """
        self.eg = eg
        self.cell_T = cell_T
        self.n_c = n_c
        self.n_s = n_s

        self._construct()

    def _construct(self):
        self.j01 = calculate_j01(self.eg, temperature=self.cell_T,
                                 n1=1, n_c=self.n_c, n_s=self.n_s)

    def set_input_spectrum(self, input_spectrum):
        self.ill = input_spectrum
        self.jsc = calc_jsc_from_eg(input_spectrum, self.eg)

    def get_transmit_spectrum(self):
        sp = self.ill.get_spectrum(to_x_unit='eV')

        filter_y = sp[0, :] < self.eg

        filter = Spectrum(sp[0, :], filter_y, x_unit='eV')

        return self.ill * filter

    def get_eta(self):
        volt = np.linspace(-0.5, self.eg, num=300)
        volt, current = gen_rec_iv_by_rad_eta(self.j01, 1, 1, self.cell_T, 1e15, voltage=volt, jsc=self.jsc)

        max_p = max_power(volt, current)

        return max_p / self.ill.total_power()

    def get_iv(self, volt=None):
        if volt == None:
            volt = np.linspace(-0.5, self.eg, num=300)

        volt, current = gen_rec_iv_by_rad_eta(self.j01, 1, 1, self.cell_T, 1e15, voltage=volt, jsc=self.jsc)

        return volt, current


class DBCell(SolarCell):
    def __init__(self, qe, rad_eta, T, n_c=3.5, n_s=1, qe_cutoff=1e-3):
        """
        Initialize the solar cell object

        :param T: temperature of the cell
        :param qe: quantum efficiency of the cell
        :type qe: Spectrum
        :param rad_eta: external radiative efficiency of the cell
        :param n_c: refractive index of the cell
        :param n_s: refractive index of ambient
        :param qe_cutoff: set the qe value to zero if it is lower than qe_cutoff. This is for avoiding the small ground in experimental data ends up becoming large when multiplying generalized Planck's distribution.
        """

        self.qe = qe
        self.rad_eta = rad_eta
        self.n_c = n_c
        self.n_s = n_s
        self.qe_cutoff = qe_cutoff
        self.cell_T = T

        self.ill = None
        self.jsc = None

        self._check_qe()
        self._construct()


    def _construct(self):

        self.j01 = calculate_j01_from_qe(self.qe, n_c=self.n_c, n_s=self.n_s, threshold=self.qe_cutoff, T=self.cell_T)

    def _check_qe(self):
        """
        Check if all the values of QE is <1 and >0
        :return:
        """

        if (np.all(self.qe.core_y>=0) and np.all(self.qe.core_y<=1))==False:
            raise ValueError("The values of QE should be between 0 and 1.")




    def set_input_spectrum(self, input_spectrum):
        self.ill = input_spectrum
        self.jsc = calc_jsc(self.ill, self.qe)

    def get_transmit_spectrum(self):
        """

        :return: the transmitted spetrum
        :rtype: Spectrum
        """

        filtered_sp = self.ill * (1 - self.qe)

        return filtered_sp

    def get_iv(self, volt=None):
        if volt is None:
            volt = np.linspace(-0.5, 5, num=300)

        volt, current = gen_rec_iv_by_rad_eta(self.j01, rad_eta=self.rad_eta, n1=1,
                                              temperature=self.cell_T, rshunt=1e15, voltage=volt, jsc=self.jsc)

        return volt, current

    def get_eta(self):

        # Guess the required limit of maximum voltage
        volt_lim = self.rad_eta * np.log(self.jsc / self.j01) * thermal_volt * self.cell_T

        volt, current = self.get_iv(volt=np.linspace(-0.5, volt_lim + 0.3, 300))

        max_p = max_power(volt, current)

        return max_p / self.ill.total_power()


class MJCell(SolarCell):
    def __init__(self, subcell):
        """

        :param subcell: A list of SolarCell instances of multijunction cell from top to bottom , e.g. [top_cell mid_cell bot_cell]
        :type subcell: List[SolarCell]
        """

        self.subcell = subcell

    def set_input_spectrum(self, input_spectrum):

        self.ill = input_spectrum
        filtered_spectrum = None

        # Set spectrum for each subcell
        for i, sc in enumerate(self.subcell):
            if i == 0:
                sc.set_input_spectrum(input_spectrum)
                filtered_spectrum = sc.get_transmit_spectrum()
            else:
                sc.set_input_spectrum(filtered_spectrum)

    def get_iv(self, volt=None):

        all_iv = [(sc.get_iv()) for sc in self.subcell]

        v, i = solve_mj_iv(all_iv, i_max=20)

        return v, i

    def get_eta(self):

        v, i = self.get_iv()

        eta = max_power(v, i)

        return eta / self.ill.total_power()