__author__ = 'kanhua'


import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from scipy.io import savemat
from iii_v_si import calc_2j_si_eta, calc_3j_si_eta
import matplotlib as mpl


font = {'size'   : 8}
mpl.rc('font', **font)



if __name__=="__main__":

    top_bg=np.linspace(1.5,2.2,num=20)
    mid_bg=np.linspace(1.2,1.6,num=20)

    eta_array=np.zeros((top_bg.shape[0],mid_bg.shape[0]))

    for i,teg in enumerate(top_bg):
        for j,meg in enumerate(mid_bg):
            eta,_,_,_= calc_3j_si_eta(1, 1, 1, top_band_gap=teg, top_cell_qe=1, mid_band_gap=meg,
                                      mid_cell_qe=1,bot_cell_eta=1,bot_band_gap=1)
            eta_array[i,j]=eta


    fig = plt.figure(figsize=(3.75, 3.5), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)

    np.savez("rad_limit_3J.npz",tbg=top_bg,mbg=mid_bg,eta=eta_array)

    cax=ax.pcolormesh(mid_bg,top_bg,eta_array)
    fig.colorbar(cax)
    ax.set_xlabel("band gap of middle cell (eV)")
    ax.set_ylabel("band gap of top cell (eV)")
    ax.set_xlim([np.min(mid_bg),np.max(mid_bg)])
    ax.set_ylim([np.min(top_bg),np.max(top_bg)])

    plt.savefig("rad_limit_3J_1eV.png",dpi=150)
    plt.show()
