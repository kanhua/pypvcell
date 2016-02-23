"""
This script tests the new detailed balance model of Si
"""

__author__ = 'kanhua'


import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from scipy.io import savemat
from iii_v_si import calc_2j_si_eta, calc_3j_si_eta


si_layer = np.logspace(-6, -2, num=100) * 1e6


gaas_eta=[calc_2j_si_eta(s * 1e-6, 1, 1.42, top_cell_qe=1, top_cell_rad_eta=1)[0] for s in si_layer]
ingap_eta=[calc_2j_si_eta(s * 1e-6, 1, 1.87, top_cell_qe=1, top_cell_rad_eta=1)[0] for s in si_layer]
ingap2_eta=[calc_2j_si_eta(s * 1e-6, 0.005, 1.87, top_cell_qe=1, top_cell_rad_eta=1)[0] for s in si_layer]


#plt.semilogx(si_layer, gaas_eta, label="GaAs (69% QE)", hold=True)
plt.semilogx(si_layer, ingap_eta, label="InGaP (100% Si RE)",hold=True)
plt.semilogx(si_layer, ingap2_eta, label="InGaP (0.005 Si RE)")
plt.legend(loc="best")
plt.grid()

plt.xlabel("thickness of Si layer (um)")
plt.ylabel("efficiency")
plt.savefig("t_dep_eta_2.pdf")

print(np.max(ingap_eta))
print(si_layer[np.argmax(ingap_eta)])








