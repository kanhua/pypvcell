__author__ = 'kanhua'

import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from scipy.io import savemat
from iii_v_si import calc_2j_si_eta, calc_3j_si_eta


if __name__=="__main__":
    algaas_top_ere=np.logspace(-7,0,num=50)
    algaas_mid_ere=np.logspace(-7,0,num=50)

    eta_array=np.zeros((algaas_top_ere.shape[0],algaas_mid_ere.shape[0]))

    for i,teg in enumerate(algaas_top_ere):
        for j,meg in enumerate(algaas_mid_ere):
            eta,_,_,_= calc_3j_si_eta(teg, meg, 1, top_band_gap=1.85, top_cell_qe=0.8, mid_band_gap=1.45,
                                      mid_cell_qe=0.8)
            eta_array[i,j]=eta


    np.savez("algaas_algaas_ere_3J.npz",tbg=algaas_top_ere,mbg=algaas_mid_ere,eta=eta_array)

    plt.pcolormesh(algaas_mid_ere,algaas_top_ere,eta_array)
    plt.colorbar()
    plt.xlabel("band gap of middle cell (eV)")
    plt.ylabel("band gap of top cell (eV)")
    plt.xlim([np.min(algaas_mid_ere),np.max(algaas_mid_ere)])
    plt.ylim([np.min(algaas_top_ere),np.max(algaas_top_ere)])
    plt.xscale("log")
    plt.yscale("log")

    plt.savefig("algaas_algaas_ere_3J.pdf")
    plt.show()