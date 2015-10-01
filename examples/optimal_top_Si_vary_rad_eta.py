"""
This script tests the new detailed balance model of Si
"""

import numpy as np
from ivsolver import calculate_j01_from_qe, gen_rec_iv, calculate_j01, calculate_j02_from_rad_eff
from detail_balanced_MJ import calc_mj_eta,rad_to_voc,extract_voc
from photocurrent import gen_qe_from_abs, gen_square_qe, calc_jsc
from spectrum_base import spectrum_base
import matplotlib.pyplot as plt
from fom import voc
from illumination import illumination, bp_filter
from scipy.interpolate import interp2d

__author__ = 'kanhua'


def calc_si_iv(si_layer_t, si_rad_eta, top_cell_bg, top_cell_qe=1, top_cell_rad_eta=1):
    si_layer = si_layer_t
    abs_file = "./si_alpha.csv"

    si_alpha = np.loadtxt(abs_file, delimiter=',')

    si_alpha_sp = spectrum_base()
    si_alpha_sp.set_spectrum(si_alpha[:, 0], si_alpha[:, 1], wavelength_unit='m')

    qe = gen_qe_from_abs(si_alpha_sp, si_layer)

    j01 = calculate_j01_from_qe(qe)

    test_voltage = np.linspace(-0.5, 1.8, num=300)

    j02 = calculate_j02_from_rad_eff(j01, si_rad_eta, test_voltage, 300, 2)

    v, i = gen_rec_iv(j01, j02, 1, 2, 300, 1e10, test_voltage)

    si_voc = extract_voc(v, i, qe)

    top_voc=rad_to_voc(top_cell_rad_eta,gen_square_qe(top_cell_bg,top_cell_qe))

    eta = calc_mj_eta([top_cell_bg, 1.1], [top_cell_qe, 1], [top_cell_rad_eta, si_rad_eta], 300, replace_iv=(1, (v, i)),
                      replace_qe=(1, qe))

    return eta, si_voc, top_voc



def plot_voc(top_voc, bot_voc, eta,top_cell_bg=1.87,bot_cell_bg=1.1):
    plt.pcolormesh(top_cell_bg - top_voc, bot_cell_bg - bot_voc, eta)

    plt.colorbar()

    # plt.legend(loc="best")
    plt.xlim([np.min(top_cell_bg-top_voc),np.max(top_cell_bg-top_voc)])
    plt.ylim([np.min(bot_cell_bg-bot_voc),np.max(bot_cell_bg-bot_voc)])
    plt.xlabel("Eg/q-Voc (1.7eV top cell)")
    plt.ylabel("Eg/q-Voc (Si bottom cell)")
    plt.savefig("voc_dep_eta.pdf")
    plt.close()


def plot_rad_eta(top_rad_eta, bot_rad_eta, eta):
    plt.pcolormesh(top_rad_eta, bot_rad_eta, eta)

    plt.xscale("log")
    plt.yscale("log")
    plt.colorbar()

    # plt.legend(loc="best")
    plt.xlabel("radiative efficiency (1.7eV top cell)")
    plt.ylabel("radiative efficiency (Si bottom cell)")
    plt.savefig("radeta_dep_eta.pdf")


si_layer_num = 100
si_rad_eta_ar_num = 50

si_layer = np.logspace(-6, -2, num=si_layer_num) * 1e6

si_rad_eta_ar = np.logspace(-7, 0, num=si_rad_eta_ar_num)

iii_v_rad_eta_ar = np.logspace(-7, 0, num=si_rad_eta_ar_num)


# gaas_eta = [calc_si_iv(s * 1e-6, 1.42, top_cell_qe=0.69) for s in si_layer]


ingap_eta_voc = [calc_si_iv(1000 * 1e-6, s, 1.7, top_cell_rad_eta=t) for s in si_rad_eta_ar for t in iii_v_rad_eta_ar]

ingap_eta = [w[0] for w in ingap_eta_voc]
si_voc = [float(w[1]) for w in ingap_eta_voc]
ingap_voc = [float(w[2]) for w in ingap_eta_voc]

si_voc = np.array(si_voc).reshape((iii_v_rad_eta_ar.shape[0], si_rad_eta_ar_num))
ingap_eta = np.array(ingap_eta).reshape((iii_v_rad_eta_ar.shape[0], si_rad_eta_ar_num))
ingap_voc = np.array(ingap_voc).reshape((iii_v_rad_eta_ar.shape[0], si_rad_eta_ar_num))


np.savez("radeta_1.7.npz",eta=ingap_eta,voc_y=si_voc,voc_x=ingap_voc)
np.savez("radeta_1.7_radeta.npz",eta=ingap_eta,ere_y=si_rad_eta_ar,ere_x=iii_v_rad_eta_ar)

# plt.semilogx(si_layer, gaas_eta, label="GaAs (69% QE)", hold=True)
# plt.semilogx(si_rad_eta_ar, ingap_eta[:,0], 'o',label="InGaP (100% QE)")
#plt.semilogx(si_rad_eta_ar, ingap_eta[:,4], 'o',label="InGaP (100% QE)")

print(ingap_eta)
#plt.pcolormesh(iii_v_rad_eta_ar, si_voc, ingap_eta)

plot_voc(ingap_voc, si_voc, ingap_eta)

plot_rad_eta(iii_v_rad_eta_ar, si_rad_eta_ar, ingap_eta)













