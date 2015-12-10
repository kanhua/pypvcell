__author__ = 'kanhua'


import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from scipy.io import savemat
from iii_v_si import calc_2j_si_eta, calc_3j_si_eta


def calc_std_3J():
    algaas_bg_top=np.linspace(1.45,2.0,num=40)
    algaas_bg_mid=np.linspace(1.0,1.8,num=40)

    eta_array=np.zeros((algaas_bg_top.shape[0],algaas_bg_mid.shape[0]))

    for i,teg in enumerate(algaas_bg_top):
        for j,meg in enumerate(algaas_bg_mid):
            eta,_,_,_= calc_3j_si_eta(1, 1, 240/0.9, top_band_gap=teg, top_cell_qe=1, mid_band_gap=meg,
                                      mid_cell_qe=1,bot_cell_qe=1,bot_band_gap=0.67,bot_cell_eta=1)
            eta_array[i,j]=eta


    np.savez("standard_3J.npz",tbg=algaas_bg_top,mbg=algaas_bg_mid,eta=eta_array)



if __name__=="__main__":

    #calc_std_3J()

    data=np.load("standard_3J.npz")
    eta_array=data["eta"]
    top_bg=data["tbg"]
    mid_bg=data["mbg"]

    max_j=np.argmax(np.max(eta_array, axis=0))
    max_i=np.argmax(np.max(eta_array, axis=1))

    assert eta_array[max_i,max_j]==np.max(eta_array)

    print("max top cell Eg:%s"%top_bg[max_i])
    print("max middle cell Eg: %s"%mid_bg[max_j])
    print("max eta: %s"%eta_array.max())


    plt.pcolormesh(mid_bg,top_bg,eta_array)
    plt.colorbar()
    plt.xlabel("band gap of middle cell (eV)")
    plt.ylabel("band gap of top cell (eV)")
    plt.xlim([np.min(mid_bg),np.max(mid_bg)])
    plt.ylim([np.min(top_bg),np.max(top_bg)])
    plt.savefig("standard_3J.pdf")
    plt.show()
