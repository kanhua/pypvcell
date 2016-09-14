"""
This script tests the new detailed balance model of Si
"""

import numpy as np
from solcore3.beta.ivsolver import calculate_j01_from_qe, gen_rec_iv, calculate_j01, calculate_j02_from_rad_eff
from solcore3.beta.detail_balanced_MJ import calc_mj_eta
from solcore3.beta.photocurrent import gen_qe_from_abs, gen_square_qe
from solcore3.beta.spectrum_base import spectrum_base
import matplotlib.pyplot as plt


__author__ = 'kanhua'


def calc_si_iv(si_layer_t, si_rad_eta, top_cell_bg, top_cell_qe=1):
    si_layer = si_layer_t
    abs_file = "../si_alpha.csv"

    si_alpha = np.loadtxt(abs_file, delimiter=',')

    si_alpha_sp = spectrum_base()
    si_alpha_sp.set_spectrum(si_alpha[:, 0], si_alpha[:, 1], x_unit='m')

    qe = gen_qe_from_abs(si_alpha_sp, si_layer)

    j01 = calculate_j01_from_qe(qe)

    test_voltage = np.linspace(-0.5, 1.5, num=300)

    j02 = calculate_j02_from_rad_eff(j01, si_rad_eta, test_voltage, 300, 2)

    v, i = gen_rec_iv(j01, j02, 1, 2, 300, 1e10, test_voltage)

    eta = calc_mj_eta([top_cell_bg, 1.1], [top_cell_qe, 1], [1, si_rad_eta], 300, replace_iv=(1, (v, i)),
                      replace_qe=(1, qe))

    return eta


si_layer_num=100
si_rad_eta_ar_num=100

si_layer = np.logspace(-6, -2, num=si_layer_num) * 1e6

si_rad_eta_ar = np.logspace(-6, 0, num=si_rad_eta_ar_num)

iii_v_rad_eta_ar=np.logspace(-5,0,num=si_rad_eta_ar_num)

# gaas_eta = [calc_si_iv(s * 1e-6, 1.42, top_cell_qe=0.69) for s in si_layer]


ingap_eta = [calc_si_iv(t*1e-6, s, 1.9) for s in si_rad_eta_ar for t in si_layer]

ingap_eta=np.array(ingap_eta).reshape((si_rad_eta_ar_num,si_layer_num))


#plt.semilogx(si_layer, gaas_eta, label="GaAs (69% QE)", hold=True)
#plt.semilogx(si_rad_eta_ar, ingap_eta[:,0], 'o',label="InGaP (100% QE)")

#plt.semilogx(si_rad_eta_ar, ingap_eta[:,4], 'o',label="InGaP (100% QE)")

print(ingap_eta)
plt.pcolormesh(si_layer,si_rad_eta_ar,ingap_eta)
plt.xscale("log")
plt.yscale("log")
plt.colorbar()

#plt.legend(loc="best")
plt.xlabel("thickness of Si (um)")
plt.ylabel("radiative efficiency")
plt.savefig("radeta_dep_eta.pdf")









