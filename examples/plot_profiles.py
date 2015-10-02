__author__ = 'kanhua'

import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from scipy.io import savemat
from iii_v_si import calc_2j_si_eta, calc_3j_si_eta
import matplotlib as mpl

mpl.rc('font', size=8)
mpl.rc('xtick', labelsize=8)
mpl.rc('ytick', labelsize=8)


def calc_2j_eta_array(iii_v_rad_eta_ar, top_cell_bg, top_cell_qe):
    eta_arr = list()
    for r in iii_v_rad_eta_ar:
        eta, _, _ = calc_2j_si_eta(si_layer_t=1000e-6, si_rad_eta=0.005, top_cell_bg=top_cell_bg,
                                   top_cell_qe=top_cell_qe, top_cell_rad_eta=r)
        eta_arr.append(eta)
    eta_arr = np.array(eta_arr)
    plt_label = "{},{}".format(top_cell_bg, top_cell_qe)
    return eta_arr, plt_label


def calc_3j_eta_array(iii_v_rad_eta, top_cell_bg, top_cell_qe, mid_cell_bg, mid_cell_qe):
    eta_arr = list()
    for r in iii_v_rad_eta_ar:
        eta, _, _, _ = calc_3j_si_eta(r, r, 1, top_band_gap=top_cell_bg, mid_band_gap=mid_cell_bg,
                                      top_cell_qe=top_cell_qe, mid_cell_qe=mid_cell_qe)
        # calc_2j_si_eta(si_layer_t=1000e-6, si_rad_eta=0.005, top_cell_bg=top_cell_bg,
        #                       top_cell_qe=top_cell_qe, top_cell_rad_eta=r)
        eta_arr.append(eta)
    eta_arr = np.array(eta_arr)
    plt_label = "{}/{},{}".format(top_cell_bg, mid_cell_bg,top_cell_qe)
    return eta_arr, plt_label

    pass


if __name__ == "__main__":
    si_rad_eta_ar_num = 10
    iii_v_rad_eta_ar = np.logspace(-7, 0, num=si_rad_eta_ar_num)
    e1, lab1 = calc_2j_eta_array(iii_v_rad_eta_ar, 1.7, 1)
    e2, lab2 = calc_2j_eta_array(iii_v_rad_eta_ar, 1.89, 1)
    e3, lab3 = calc_2j_eta_array(iii_v_rad_eta_ar, 1.7, 0.8)

    e4, lab4 = calc_3j_eta_array(iii_v_rad_eta_ar, 1.87, 1, 1.42, 1)
    e6, lab6 = calc_3j_eta_array(iii_v_rad_eta_ar, 1.87, 0.8, 1.42, 0.8)
    e5, lab5 = calc_3j_eta_array(iii_v_rad_eta_ar, 1.97, 1, 1.48, 1)

    fig = plt.figure(figsize=(3.75, 3.5), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)
    ax.semilogx(iii_v_rad_eta_ar, e1, label=lab1)
    ax.semilogx(iii_v_rad_eta_ar, e2, label=lab2)
    ax.semilogx(iii_v_rad_eta_ar, e3, label=lab3)
    ax.semilogx(iii_v_rad_eta_ar, e4, label=lab4)
    ax.semilogx(iii_v_rad_eta_ar, e5, label=lab5)
    ax.semilogx(iii_v_rad_eta_ar,e6,label=lab6)
    ax.legend(loc='best')

    ax.set_xlabel("ERE of top cell")
    ax.set_ylabel("efficiency")
    ax.grid(True)

    fig.savefig('myfig.png', dpi=300)

    fig.show()




