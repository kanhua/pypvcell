"""
This scripts performs the calculation on the paper "Efficiency calculations
of thin-film GaAs solar cells on Si substrates" by M. Yamaguchi, JAP 58,3601
papers3://publication/doi/10.1063/1.335737
"""
__author__ = 'kanhua'

import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
from detail_balanced_MJ import calc_mj_eta

def rad_eff(Nd):
    L_0=1e-3
    return 1/(1+np.power(L_0,2)*np.power(sc.pi,3)*Nd/4)

def check_rad_eff():
    nd=np.logspace(3,6,num=100)
    plt.semilogx(nd,rad_eff(nd))
    plt.show()

def calc_4j_eta(top_band_gap, top_band_dislocation, concentration):
    cell_temperature = 300
    subcell_eg = np.array([top_band_gap, 1.12])
    subcell_qe = np.array([0.69, 1])
    subcell_rad_eff = np.array([1, rad_eff(top_band_dislocation)])

    return calc_mj_eta(subcell_eg, subcell_qe, subcell_rad_eff, cell_temperature, concentration=concentration)


test_nd=np.logspace(3,6,num=100)

eta_mtx=[calc_4j_eta(1.42,nd,1) for nd in test_nd]

plt.semilogx(test_nd,eta_mtx)
plt.xlabel("dislocation density")
plt.ylabel("efficiency")
plt.savefig("GaAs on Si.pdf")
plt.show()

