__author__ = 'kanhua'


import numpy as np
from ivsolver import calculate_j01_from_qe, gen_rec_iv, calculate_j01, calculate_j02_from_rad_eff
from detail_balanced_MJ import calc_mj_eta, rad_to_voc, extract_voc
from photocurrent import gen_qe_from_abs, gen_square_qe, calc_jsc
from spectrum_base import spectrum_base
import matplotlib.pyplot as plt
from fom import voc
from illumination import illumination, bp_filter
from scipy.interpolate import interp2d


if __name__=="__main__":


    cell_temperature=300
    subcell_eg=[2.0,1.42,1.12,0.5]
    subcell_rad_eff=[1e-3,1e-2,1e-3,1e-3]
    subcell_qe=[0.8,0.9,0.9,0.8]
    concentration=1

    top_cell_bg=np.linspace(1.5,2.2,num=30)
    bot_cell_bg=np.linspace(0.5,1.1,num=30)


    eta_array=np.zeros((top_cell_bg.shape[0],bot_cell_bg.shape[0]))
    for i,t in enumerate(top_cell_bg):
        for j,b in enumerate(bot_cell_bg):

            subcell_eg[0]=t
            subcell_eg[3]=b
            eta_array[i,j]= calc_mj_eta(subcell_eg, subcell_qe, subcell_rad_eff, cell_temperature,
                                        concentration=concentration)

    maxidx_0=eta_array.argmax(axis=0)
    maxidx_1=eta_array[maxidx_0].argmax()

    print(top_cell_bg[maxidx_0[maxidx_1]])
    print(bot_cell_bg[maxidx_1])

    print(eta_array.max())

    plt.pcolormesh(bot_cell_bg,top_cell_bg,eta_array)
    plt.xlim([np.min(bot_cell_bg),np.max(bot_cell_bg)])
    plt.ylim([np.min(top_cell_bg),np.max(top_cell_bg)])
    plt.xlabel("band gap of material Y")
    plt.ylabel("band gap of material X")
    plt.colorbar()
    plt.savefig("x_GaAs_Si_y.pdf")

    plt.show()




