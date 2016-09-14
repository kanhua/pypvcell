__author__ = 'kanhua'

import numpy as np
from solcore3.beta.ivsolver import calculate_j01_from_qe, gen_rec_iv, calculate_j01, calculate_j02_from_rad_eff
from solcore3.beta.detail_balanced_MJ import calc_mj_eta
from solcore3.beta.photocurrent import gen_qe_from_abs, gen_square_qe,calc_jsc
from solcore3.beta.spectrum_base import spectrum_base
import matplotlib.pyplot as plt
from solcore3.beta.fom import voc
from solcore3.beta.illumination import illumination,bp_filter


__author__ = 'kanhua'


def calc_si_iv(si_layer_t, si_rad_eta,ingap_filtered=False):
    si_layer = si_layer_t
    abs_file = "../si_alpha.csv"

    si_alpha = np.loadtxt(abs_file, delimiter=',')

    si_alpha_sp = spectrum_base()
    si_alpha_sp.set_spectrum(si_alpha[:, 0], si_alpha[:, 1], x_unit='m')

    qe = gen_qe_from_abs(si_alpha_sp, si_layer)

    j01 = calculate_j01_from_qe(qe)

    test_voltage = np.linspace(-0.5, 1.5, num=300)

    j02 = calculate_j02_from_rad_eff(j01, si_rad_eta, test_voltage, 300, 2)


    input_ill = illumination(concentration=1)

    if ingap_filtered:
        bf=bp_filter(1.9)
        input_ill=input_ill.attenuation([bf])

    jsc=calc_jsc(input_ill,qe)

    v, i = gen_rec_iv(j01, j02, 1, 2, 300, 1e10, test_voltage,jsc=jsc)


    return v,i


if __name__=="__main__":

    rad_arr=np.logspace(-6,0,num=100)
    voc_arr=[]
    voc_ingap_filtered_arr=[]

    for r in rad_arr:
        v,i=calc_si_iv(100e-6,r)
        voc_arr.append(voc(v,i))


        v,i=calc_si_iv(100e-6,r,ingap_filtered=True)
        voc_ingap_filtered_arr.append(voc(v,i))

    plt.semilogx(rad_arr,np.array(voc_arr),hold=True,label="1-sun")
    plt.semilogx(rad_arr,np.array(voc_ingap_filtered_arr),label="InGaP filtered 1-sun")
    plt.xlabel("radiative efficiency")
    plt.ylabel("Voc of Si cell")
    plt.grid()
    plt.legend(loc="best")
    plt.savefig("rad_eta_voc.pdf")

    plt.show()

