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

    topmid_radeta=np.logspace(-5,0,num=50)
    bot_rad_eta=np.logspace(-6,0,num=50)

    eta_array=np.zeros((topmid_radeta.shape[0],bot_rad_eta.shape[0]))

    for i,teg in enumerate(topmid_radeta):
        for j,meg in enumerate(bot_rad_eta):
            eta,_,_,_= calc_3j_si_eta(teg, teg, 1, top_band_gap=1.97, top_cell_qe=1, mid_band_gap=1.48,
                                      mid_cell_qe=1,bot_cell_eta=meg,bot_band_gap=1.12)
            eta_array[i,j]=eta


    fig = plt.figure(figsize=(3.75, 3.5), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)

    np.savez("rad_eta_3J.npz",tbg=topmid_radeta,mbg=bot_rad_eta,eta=eta_array)

    cax=ax.pcolormesh(bot_rad_eta,topmid_radeta,eta_array)
    fig.colorbar(cax)
    ax.set_xlabel("ERE of silicon cell (eV)")
    ax.set_ylabel("ERE of top and middle cell")
    ax.set_xlim([np.min(bot_rad_eta),np.max(bot_rad_eta)])
    ax.set_ylim([np.min(topmid_radeta),np.max(topmid_radeta)])
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.savefig("rad_eta_3J.png",dpi=100)
    plt.show()
