"""
detalied balance calculation of III-V on Si

"""

import matplotlib.pyplot as plt
import numpy as np
from detail_balanced_MJ import calc_mj_eta,rad_to_voc,extract_voc
from photocurrent import gen_qe_from_abs, gen_square_qe, calc_jsc
import yaml


top_bg=1.97
mid_bg=1.48

def calc_4j_eta(top_cell_eta,mid_cell_eta, concentration,top_band_gap=1.87, mid_band_gap=1.42) :

    si_bg=1.12
    cell_temperature = 300
    subcell_eg = np.array([top_band_gap, mid_band_gap,si_bg])
    subcell_qe = np.array([1, 1, 1])
    subcell_rad_eff = np.array([top_cell_eta, top_cell_eta,mid_cell_eta])


    top_voc=rad_to_voc(top_cell_eta,gen_square_qe(top_band_gap,subcell_qe[0]))
    mid_voc=rad_to_voc(top_cell_eta,gen_square_qe(mid_band_gap,subcell_qe[1]))
    bot_voc=rad_to_voc(mid_cell_eta,gen_square_qe(si_bg,subcell_qe[2]))


    return calc_mj_eta(subcell_eg,subcell_qe,subcell_rad_eff,cell_temperature,concentration=concentration),\
           top_voc,mid_voc,bot_voc

test_top_eta=np.logspace(-7,0,num=40,endpoint=True)
test_mid_eta=np.logspace(-7,0,num=40,endpoint=True)

eta_array_1eV_1sun = [calc_4j_eta(eg, eg2, concentration=1,top_band_gap=top_bg,mid_band_gap=mid_bg) for eg2 in test_mid_eta for eg in test_top_eta]

eta_array,voc1,voc2,voc3=zip(*eta_array_1eV_1sun)

eta_mtx=np.array(eta_array).reshape((len(test_mid_eta),len(test_top_eta)))

voc1_mtx=np.array(voc1).reshape((len(test_mid_eta),len(test_top_eta)))
voc2_mtx=np.array(voc2).reshape((len(test_mid_eta),len(test_top_eta)))
voc3_mtx=np.array(voc3).reshape((len(test_mid_eta),len(test_top_eta)))


np.savez('mjdata_opt.npz',voc_x=voc1_mtx,voc_y=voc2_mtx,voc_z=voc3_mtx,eta=eta_mtx)

#yaml.dump([test_top_eta,test_mid_eta,eta_mtx],open("III-V_si_3J_eta.yaml",'w'))


plt.pcolormesh(test_top_eta,test_mid_eta,eta_mtx)
plt.xscale("log")
plt.yscale("log")

plt.ylabel("radiative efficiency (Si bottom cell)")
plt.xlabel("radiative efficiency (top and mid cell)")
plt.colorbar()
plt.savefig("rad_eta_3J_low.pdf")
plt.show()