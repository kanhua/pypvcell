__author__ = 'kanhua'

import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from scipy.io import savemat
from iii_v_si import calc_2j_si_eta, calc_2j_si_eta_direct



def calc_gaas(n_s=1):
    GaAs_eg = 1.42

    topcell_eqe = np.linspace(0.6, 1, num=50)
    eta = np.zeros(topcell_eqe.shape)

    for p in range(topcell_eqe.shape[0]):
        # eta[p], _, _ = calc_2j_si_eta(1000e-6, 1, GaAs_eg, top_cell_qe=topcell_eqe[p], top_cell_rad_eta=1,
        #                              spectrum="AM1.5g")
        eta[p], _, _ = calc_2j_si_eta_direct(top_eg=GaAs_eg, top_rad_eta=1, top_qe=topcell_eqe[p], bot_rad_eta=1,
                                             bot_qe=1, n_s=n_s)

    print("At AM1.5g, direct band gap assumption of silicon")
    print("gaas max eta %s:" % eta.max())
    print("optimal EQE for GaAs: %s" % topcell_eqe[eta.argmax()])


if __name__=="__main__":
    calc_gaas(n_s=1)
    calc_gaas(n_s=3.5)

