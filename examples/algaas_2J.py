__author__ = 'kanhua'

import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from scipy.io import savemat
from iii_v_si import calc_2j_si_eta, calc_3j_si_eta


if __name__=="__main__":
    algaas_ere=np.logspace(-7,0,num=50)
    algaas_bg=np.linspace(1.45,2,num=50)

    eta_array=np.zeros((algaas_ere.shape[0],algaas_bg.shape[0]))

    for i,r in enumerate(algaas_ere):
        for j,bg in enumerate(algaas_bg):
            eta,_,_= calc_2j_si_eta(1000e-6, 0.005, bg, top_cell_qe=0.8, top_cell_rad_eta=r)
            eta_array[i,j]=eta


    np.savez("algaas2J.npz",bg=algaas_bg,ere=algaas_ere,eta=eta_array)

    plt.pcolormesh(algaas_bg,algaas_ere,eta_array)
    plt.colorbar()
    plt.xlabel("band gap of top cell (eV)")
    plt.yscale("log")
    plt.ylabel("external radiative efficiency")
    plt.xlim([np.min(algaas_bg),np.max(algaas_bg)])

    plt.savefig("algaas_2J.pdf")
    plt.show()










