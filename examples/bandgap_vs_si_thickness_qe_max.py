__author__ = 'kanhua'


import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from scipy.io import savemat
from iii_v_si import calc_2j_si_eta, calc_3j_si_eta

def calculate():
    si_layer = np.logspace(-6, -2, num=30) * 1e6
    top_cell_gap=np.linspace(1.4,2.2,num=30)
    qe_range=np.linspace(0.6,1,num=10)

    eta_array=np.zeros((si_layer.shape[0],top_cell_gap.shape[0]))

    for i, bg in enumerate(top_cell_gap):
        for j, t in enumerate(si_layer):
            test_eta_arr=[calc_2j_si_eta(t*1e-6, 1, bg, top_cell_qe=q, top_cell_rad_eta=1)[0] for q in qe_range]
            eta_array[i][j]= np.max(test_eta_arr)

    np.savez("bg_vs_t_qe_opt.npz",tbg=si_layer,mbg=top_cell_gap,eta=eta_array)

calculate()

#fig = plt.figure(figsize=(2.5, 2.5*3.5/3.75), tight_layout=True)
data=np.load("bg_vs_t_qe_opt.npz")
eta_array=data["eta"]
top_cell_gap=data["mbg"]
si_layer=data["tbg"]
#fig = plt.figure(figsize=(2.5, 2.5*3.5/3.75), tight_layout=True)
fig=plt.figure(tight_layout=True)
ax = fig.add_subplot(111)


cax=ax.pcolormesh(si_layer,top_cell_gap,eta_array)
ax.set_xscale("log")
ax.set_ylabel("band gap of top cell")
ax.set_xlabel("thickness of silicon")
ax.set_ylim([np.min(top_cell_gap),np.max(top_cell_gap)])
ax.set_xlim([np.min(si_layer),np.max(si_layer)])
fig.colorbar(cax)

plt.show()

