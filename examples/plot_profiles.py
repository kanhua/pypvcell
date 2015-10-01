__author__ = 'kanhua'



import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from scipy.io import savemat
from iii_v_si import calc_2j_si_eta
import matplotlib as mpl
mpl.rc('font',size=10)
mpl.rc('xtick', labelsize=10)
mpl.rc('ytick', labelsize=10)


def calc_eta_array(iii_v_rad_eta_ar,top_cell_bg,top_cell_qe):
    eta_arr=list()
    for r in iii_v_rad_eta_ar:
        eta,_,_=calc_2j_si_eta(si_layer_t=1000e-6, si_rad_eta=0.005, top_cell_bg=top_cell_bg, top_cell_qe=top_cell_qe, top_cell_rad_eta=r)
        eta_arr.append(eta)
    eta_arr=np.array(eta_arr)
    plt_label="Eg={},QE={}".format(top_cell_bg,top_cell_qe)
    return eta_arr,plt_label


if __name__=="__main__":

    si_rad_eta_ar_num=10
    iii_v_rad_eta_ar = np.logspace(-7, 0, num=si_rad_eta_ar_num)
    e1,lab1=calc_eta_array(iii_v_rad_eta_ar,1.7,1)
    e2,lab2=calc_eta_array(iii_v_rad_eta_ar,1.89,1)
    e3,lab3=calc_eta_array(iii_v_rad_eta_ar,1.7,0.8)


    fig=plt.figure(figsize=(3.75, 3.5), dpi=100,tight_layout=True)
    ax=fig.add_subplot(111)
    ax.semilogx(iii_v_rad_eta_ar,e1,label=lab1)
    ax.semilogx(iii_v_rad_eta_ar,e2,label=lab2)
    ax.semilogx(iii_v_rad_eta_ar,e3,label=lab3)
    ax.legend(loc='best')

    ax.set_xlabel("ERE of top cell")
    ax.set_ylabel("efficiency")

    fig.savefig('myfig.png', dpi=300)

    fig.show()




