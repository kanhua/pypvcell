from typing import List
from pypvcell.illumination import Illumination
from pypvcell.photocurrent import gen_square_qe, calc_jsc_from_eg
from pypvcell.ivsolver import calculate_j01, gen_rec_iv_by_rad_eta, solve_mj_iv
from pypvcell.fom import max_power
from pypvcell.spectrum import Spectrum
import numpy as np


class SolarCell(object):
    def __init__(self):
        pass

    def get_iv(self):
        raise NotImplementedError()

    def set_input_spectrum(self, input_spectrum: Illumination):
        raise NotImplementedError()

    def get_transmit_spectrum(self):
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

    def set_input_spectrum(self, input_spectrum: Illumination):
        self.ill = input_spectrum
        self.jsc = calc_jsc_from_eg(input_spectrum, self.eg)

    def get_transmit_spectrum(self) -> Illumination:
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


class MJCell(SolarCell):
    def __init__(self, subcell: List[SolarCell]):

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
