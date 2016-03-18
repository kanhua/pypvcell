__author__ = 'kanhua'

import numpy as np
from ivsolver import calculate_j01_from_qe, gen_rec_iv, calculate_j01, calculate_j02_from_rad_eff
from detail_balanced_MJ import calc_mj_eta, rad_to_voc, extract_voc, rad_to_voc_fast
from photocurrent import gen_qe_from_abs, gen_square_qe, calc_jsc
# from spectrum_base import spectrum_base
from spectrum_base_update import Spectrum
import matplotlib.pyplot as plt
from fom import voc
from illumination import illumination, bp_filter
from scipy.interpolate import interp2d


def calc_2j_si_eta(si_layer_t, si_rad_eta, top_cell_bg, top_cell_qe=1, top_cell_rad_eta=1, spectrum="AM1.5g", mj="2T"):
    """
    Calculate the efficiency of dual-junction solar cell with silicon bottom cell
    :param mj:
    :param spectrum:
    :param si_layer_t: thickness of silicon cell in (m)
    :param si_rad_eta:
    :param top_cell_bg:
    :param top_cell_qe:
    :param top_cell_rad_eta:
    :return: efficiency, voc of silicon cell, voc of bottom cell
    """
    si_layer = si_layer_t
    abs_file = "/Users/kanhua/Dropbox/DDocuments/Programming projects/Python/pypvcell/examples/si_alpha.csv"

    si_alpha = np.loadtxt(abs_file, delimiter=',')

    si_alpha_sp = Spectrum(wavelength=si_alpha[:, 0], spectrum=si_alpha[:, 1], wavelength_unit='m')

    qe = gen_qe_from_abs(si_alpha_sp, si_layer)

    j01 = calculate_j01_from_qe(qe)

    test_voltage = np.linspace(-0.5, 1.8, num=300)

    j02 = calculate_j02_from_rad_eff(j01, si_rad_eta, test_voltage, 300, 2)

    v, i = gen_rec_iv(j01, j02, 1, 2, 300, 1e10, test_voltage)

    si_voc = extract_voc(v, i, qe)

    top_voc = rad_to_voc(top_cell_rad_eta, gen_square_qe(top_cell_bg, top_cell_qe))

    eta = calc_mj_eta([top_cell_bg, 1.1], [top_cell_qe, 1], [top_cell_rad_eta, si_rad_eta], 300, replace_iv=(1, (v, i)),
                      replace_qe=(1, qe), spectrum=spectrum,mj=mj)

    return eta, si_voc, top_voc


def calc_2j_si_eta_direct(top_eg=1.87, top_rad_eta=1, top_qe=1, bot_eg=1.12, bot_rad_eta=0.005, bot_qe=1,
                          spectrum="AM1.5g", concentration=1, n_s=1, mj="2T", get_voc=False):
    """
    Calculate the the efficiency of 2J III-V/Si solar cell. This calculation assumes direct band gap in silicon.

    :param get_voc:
    :param mj:
    :param top_eg:
    :param top_rad_eta:
    :param top_qe:
    :param bot_eg:
    :param bot_rad_eta:
    :param bot_qe:
    :param spectrum:
    :param concentration:
    :return:
    """
    si_bg = bot_eg
    cell_temperature = 300
    subcell_eg = np.array([top_eg, si_bg])
    subcell_qe = np.array([top_qe, bot_qe])
    subcell_rad_eff = np.array([top_rad_eta, bot_rad_eta])

    top_voc = None
    bot_voc = None

    eta = calc_mj_eta(subcell_eg, subcell_qe, subcell_rad_eff, cell_temperature, concentration=concentration,
                      spectrum=spectrum, n_s=n_s, mj=mj)

    if get_voc == True:
        # top_voc = rad_to_voc(top_rad_eta, gen_square_qe(top_eg, subcell_qe[0]), max_voltage=subcell_eg[0])

        top_voc = rad_to_voc_fast(top_rad_eta, gen_square_qe(top_eg, subcell_qe[0]))
        bot_voc = rad_to_voc_fast(bot_rad_eta, gen_square_qe(si_bg, subcell_qe[1]))
        return eta, top_voc, bot_voc
    else:
        return eta



def calc_3j_si_eta(top_cell_eta, mid_cell_eta, concentration, top_band_gap=1.87, top_cell_qe=1, mid_band_gap=1.42,
                   mid_cell_qe=1, bot_cell_eta=0.005, bot_cell_qe=1, bot_band_gap=1.12, spectrum="AM1.5g"):
    si_bg = bot_band_gap
    cell_temperature = 300
    subcell_eg = np.array([top_band_gap, mid_band_gap, si_bg])
    subcell_qe = np.array([top_cell_qe, mid_cell_qe, bot_cell_qe])
    subcell_rad_eff = np.array([top_cell_eta, mid_cell_eta, bot_cell_eta])

    #top_voc = rad_to_voc(top_cell_eta, gen_square_qe(top_band_gap, subcell_qe[0]), max_voltage=subcell_eg[0])
    #mid_voc = rad_to_voc(top_cell_eta, gen_square_qe(mid_band_gap, subcell_qe[1]), max_voltage=subcell_eg[1])
    #bot_voc = rad_to_voc(mid_cell_eta, gen_square_qe(si_bg, subcell_qe[2]), max_voltage=subcell_eg[2])

    return calc_mj_eta(subcell_eg, subcell_qe, subcell_rad_eff, cell_temperature, concentration=concentration,
                       spectrum=spectrum)


def calc_gaas(n_s=1):
    GaAs_eg = 1.42

    topcell_eqe = np.linspace(0.6, 1, num=50)
    eta = np.zeros(topcell_eqe.shape)

    for p in range(topcell_eqe.shape[0]):
        # eta[p], _, _ = calc_2j_si_eta(1000e-6, 1, GaAs_eg, top_cell_qe=topcell_eqe[p], top_cell_rad_eta=1,
        #                              spectrum="AM1.5g")
        eta[p] = calc_2j_si_eta_direct(top_eg=GaAs_eg, top_rad_eta=1, top_qe=topcell_eqe[p], bot_rad_eta=1,
                                       bot_qe=1, n_s=n_s)

    print("At AM1.5g, direct band gap assumption of silicon")
    print("gaas max eta %s:" % eta.max())
    print("optimal EQE for GaAs: %s" % topcell_eqe[eta.argmax()])
    return eta.max(), topcell_eqe[eta.argmax()]
