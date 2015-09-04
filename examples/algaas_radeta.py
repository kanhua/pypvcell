__author__ = 'kanhua'

import numpy as np
from detail_balanced_MJ import rad_to_voc
from photocurrent import gen_square_qe
import matplotlib.pyplot as plt
from algaas import band_gap


def calc_algaas_qe(comp_x, qe_in_ratio):
    """
    Generate square QE of AlxGa(1-x)As
    :param comp_x:
    :param qe_in_ratio:
    :return:
    """
    bg = band_gap(comp_x)

    qe = gen_square_qe(bg, qe_in_ratio)

    return qe


if __name__ == "__main__":
    rad_eta_arr = np.logspace(-6, 0, num=100)

    qe = calc_algaas_qe(0.22, 0.8)

    voc_arr = np.array([rad_to_voc(r, qe) for r in rad_eta_arr])

    plt.plot(rad_eta_arr, voc_arr)

    plt.semilogx()
    plt.show()
