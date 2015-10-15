__author__ = 'kanhua'


import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from scipy.io import savemat
from iii_v_si import calc_2j_si_eta, calc_3j_si_eta
import matplotlib as mpl


if __name__=="__main__":


    top_eqe=np.linspace(0.7,1,num=50)

    eta_arr=list()
    for t_eqe in top_eqe:
        eta,_,_,_= calc_3j_si_eta(1, 1, 1, top_band_gap=1.87, top_cell_qe=t_eqe, mid_band_gap=1.42,
                                  mid_cell_qe=t_eqe,bot_cell_eta=1,bot_band_gap=1.12)
        eta_arr.append(eta)


    eta_arr2=list()
    for t_eqe in top_eqe:
        eta,_,_,_= calc_3j_si_eta(0.01, 0.01, 1, top_band_gap=1.87, top_cell_qe=t_eqe, mid_band_gap=1.42,
                                  mid_cell_qe=t_eqe,bot_cell_eta=0.005,bot_band_gap=1.12)
        eta_arr2.append(eta)


    plt.plot(top_eqe,eta_arr,hold=True,label="boteta=1")
    plt.plot(top_eqe,eta_arr2,hold=True,label="boteta=0.1")

    plt.show()