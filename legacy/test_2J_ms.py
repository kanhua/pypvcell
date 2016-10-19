"""
Test 2J mechanical stack cell
"""

__author__ = 'kanhua'


import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from scipy.io import savemat
from iii_v_si import calc_2j_si_eta, calc_2j_si_eta_direct



def calc_gaas(n_s=1):
    GaAs_eg = 1.2

    topcell_eqe = np.linspace(0.01, 1, num=5)
    eta = np.zeros(topcell_eqe.shape)

    for p in range(topcell_eqe.shape[0]):
        # eta[p], _, _ = calc_2j_si_eta(1000e-6, 1, GaAs_eg, top_cell_qe=topcell_eqe[p], top_cell_rad_eta=1,
        #                              spectrum="AM1.5g")
        eta[p] = calc_2j_si_eta_direct(top_eg=GaAs_eg, top_rad_eta=1, top_qe=topcell_eqe[p], bot_rad_eta=1, bot_qe=1,
                                       n_s=n_s, mj="MS")


    plt.plot(topcell_eqe,eta)
    plt.show()

    print("At AM1.5g, direct band gap assumption of silicon")
    print("gaas max eta %s:" % eta.max())
    print("optimal EQE for GaAs: %s" % topcell_eqe[eta.argmax()])


def vary_top_eg(n_s=1):


    topcell_eg = np.linspace(0.9, 3, num=100)
    eta = np.zeros(topcell_eg.shape)

    for p in range(topcell_eg.shape[0]):
        # eta[p], _, _ = calc_2j_si_eta(1000e-6, 1, GaAs_eg, top_cell_qe=topcell_eqe[p], top_cell_rad_eta=1,
        #                              spectrum="AM1.5g")
        eta[p] = calc_2j_si_eta_direct(top_eg=topcell_eg[p], top_rad_eta=1, top_qe=0.01, bot_rad_eta=1, bot_qe=1,
                                       n_s=n_s, mj="MS")


    plt.plot(topcell_eg,eta)
    plt.grid()
    plt.show()

    print("At AM1.5g, direct band gap assumption of silicon")
    print("gaas max eta %s:" % eta.max())
    print("optimal Eg: %s" % topcell_eg[eta.argmax()])


if __name__=="__main__":

    #calc_gaas(n_s=1)
    vary_top_eg()

    #calc_gaas(n_s=3.5)

