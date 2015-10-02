__author__ = 'kanhua'

import numpy as np
from ivsolver import calculate_j01_from_qe, gen_rec_iv, calculate_j01, calculate_j02_from_rad_eff
from detail_balanced_MJ import calc_mj_eta, rad_to_voc, extract_voc
from photocurrent import gen_qe_from_abs, gen_square_qe, calc_jsc
from spectrum_base import spectrum_base
import matplotlib.pyplot as plt
from fom import voc
from illumination import illumination, bp_filter
from scipy.interpolate import interp2d


def calc_2j_si_eta(si_layer_t, si_rad_eta, top_cell_bg, top_cell_qe=1, top_cell_rad_eta=1):
    """
    Calculate the efficiency of dual-junction solar cell with silicon bottom cell
    :param si_layer_t: thickness of silicon cell in (m)
    :param si_rad_eta:
    :param top_cell_bg:
    :param top_cell_qe:
    :param top_cell_rad_eta:
    :return: efficiency, voc of silicon cell, voc of bottom cell
    """
    si_layer = si_layer_t
    abs_file = "/Users/kanhua/Dropbox/Documents in Dropbox/Programming projects/Python/pypvcell/examples/si_alpha.csv"

    si_alpha = np.loadtxt(abs_file, delimiter=',')

    si_alpha_sp = spectrum_base()
    si_alpha_sp.set_spectrum(si_alpha[:, 0], si_alpha[:, 1], wavelength_unit='m')

    qe = gen_qe_from_abs(si_alpha_sp, si_layer)

    j01 = calculate_j01_from_qe(qe)

    test_voltage = np.linspace(-0.5, 1.8, num=300)

    j02 = calculate_j02_from_rad_eff(j01, si_rad_eta, test_voltage, 300, 2)

    v, i = gen_rec_iv(j01, j02, 1, 2, 300, 1e10, test_voltage)

    si_voc = extract_voc(v, i, qe)

    top_voc = rad_to_voc(top_cell_rad_eta, gen_square_qe(top_cell_bg, top_cell_qe))

    eta = calc_mj_eta([top_cell_bg, 1.1], [top_cell_qe, 1], [top_cell_rad_eta, si_rad_eta], 300, replace_iv=(1, (v, i)),
                      replace_qe=(1, qe))

    return eta, si_voc, top_voc


def calc_3j_si_eta(top_cell_eta, mid_cell_eta, concentration,
                   top_band_gap=1.87, top_cell_qe=1, mid_band_gap=1.42, mid_cell_qe=1):
    si_bg = 1.12
    cell_temperature = 300
    subcell_eg = np.array([top_band_gap, mid_band_gap, si_bg])
    subcell_qe = np.array([top_cell_qe, mid_cell_qe, 1])
    subcell_rad_eff = np.array([top_cell_eta, top_cell_eta, mid_cell_eta])

    top_voc = rad_to_voc(top_cell_eta, gen_square_qe(top_band_gap, subcell_qe[0]))
    mid_voc = rad_to_voc(top_cell_eta, gen_square_qe(mid_band_gap, subcell_qe[1]))
    bot_voc = rad_to_voc(mid_cell_eta, gen_square_qe(si_bg, subcell_qe[2]))

    return calc_mj_eta(subcell_eg, subcell_qe, subcell_rad_eff, cell_temperature, concentration=concentration), \
           top_voc, mid_voc, bot_voc