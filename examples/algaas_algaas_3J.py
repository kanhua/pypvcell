__author__ = 'kanhua'


import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from scipy.io import savemat
from iii_v_si import calc_2j_si_eta, calc_3j_si_eta


if __name__=="__main__":
    algaas_bg_top=np.linspace(1.45,2.0,num=50)
    algaas_bg_mid=np.linspace(1.41,1.8,num=50)

    eta_array=np.zeros((algaas_bg_top.shape[0],algaas_bg_mid.shape[0]))

    for i,teg in enumerate(algaas_bg_top):
        for j,meg in enumerate(algaas_bg_mid):
            eta,_,_,_= calc_3j_si_eta(1e-4, 1e-4, 1, top_band_gap=teg, top_cell_qe=0.8, mid_band_gap=meg,
                                      mid_cell_qe=0.8)
            eta_array[i,j]=eta


    np.savez("algaas_algaas_3J.npz",tbg=algaas_bg_top,mbg=algaas_bg_mid,eta=eta_array)

    plt.pcolormesh(algaas_bg_mid,algaas_bg_top,eta_array)
    plt.colorbar()
    plt.xlabel("band gap of middle cell (eV)")
    plt.ylabel("band gap of top cell (eV)")
    plt.xlim([np.min(algaas_bg_mid),np.max(algaas_bg_mid)])
    plt.ylim([np.min(algaas_bg_top),np.max(algaas_bg_top)])

    plt.savefig("algaas_algaas_3J.pdf")
    plt.show()
