"""
detalied balance calculation of many junction devices
"""

import numpy as np
import copy
from illumination import qe_filter, illumination
from fom import voc
from ivsolver import calculate_j01, calculate_j02_from_rad_eff, \
    gen_rec_iv, gen_rec_iv_with_rs_by_newton, solve_mj_iv, \
    calculate_j01_from_qe
from fom import max_power
from photocurrent import gen_square_qe, calc_jsc
import scipy.constants as sc


def set_subcell_spectrum(input_ill, subcell_eg, subcell_filter):
    subcell_ill = list()
    for layer_num, _ in enumerate(subcell_eg):

        tmp_ill = copy.deepcopy(input_ill)
        filter_set = list()
        for filter_index in range(layer_num):
            filter_set.append(subcell_filter[filter_index])
        ill = tmp_ill.attenuation(filter_set)

        subcell_ill.append(ill)
    return subcell_ill


def rad_to_voc(rad_eta, qe):
    test_voltage = np.linspace(-0.5, 1.8, num=300)

    j01_t = calculate_j01_from_qe(qe)

    j02_t = calculate_j02_from_rad_eff(j01_t, rad_eta, test_voltage, 300, n2=2)

    v_top, i_top = gen_rec_iv(j01_t, j02_t, 1, 2, 300, 1e10, test_voltage)

    top_voc = extract_voc(v_top, i_top, qe)

    return top_voc


def extract_voc(voltage, current, qe):
    input_ill = illumination(concentration=1)
    jsc = calc_jsc(input_ill, qe=qe)

    gen_current = current - jsc

    return voc(voltage, gen_current)


def calc_ere(qe, voc, T=300, ill=illumination("AM1.5g", concentration=1)):
    """
    Calculate external radiative efficiency based on Martin Green's paper
    [1]	M. A. Green, “Radiative efficiency of state-of-the-art photovoltaic cells,”
    Prog. Photovolt: Res. Appl., vol. 20, no. 4, pp. 472–476, Sep. 2011.
    :param qe: input EQE, a spectrum_base object
    :param voc: Voc of the test cell
    :param T: test tempearture of the cell, default is 300 K
    :param ill: illumination object, default is AM1.5d@1x
    :return: the calculated value of ERE
    """

    jsc = calc_jsc(ill, qe)

    jd = calculate_j01_from_qe(qe, lead_term=None)

    ere = np.exp(sc.e * voc / (sc.k * T)) * jd / jsc / (3.5 ** 2 * 2)

    return ere


def calc_mj_eta(subcell_eg, subcell_qe, subcell_rad_eff, cell_temperature, concentration=1,
                rs=0, replace_iv=None, replace_qe=None):
    subcell_eg = np.array(subcell_eg)
    subcell_qe = np.array(subcell_qe)
    subcell_rad_eff = np.array(subcell_rad_eff)

    subcell_voltage = np.linspace(-0.5, 1.9, num=300)


    # calculate j01 and j02 for each subcell
    subcell_j01 = calculate_j01(subcell_eg, cell_temperature, 1)

    subcell_j02 = [calculate_j02_from_rad_eff(subcell_j01[i], subcell_rad_eff[i], subcell_voltage, cell_temperature,
                                              2) for i, _ in enumerate(subcell_eg)]

    subcell_qe = [gen_square_qe(subcell_eg[i], subcell_qe[i]) for i, _ in enumerate(subcell_eg)]

    if replace_qe != None:
        subcell_qe[replace_qe[0]] = replace_qe[1]


    # calculate photocurrent for each subcell

    input_ill = illumination(concentration=concentration)
    subcell_filter = [qe_filter(qe.core_wl, qe.core_spec, 'm') for qe in subcell_qe]

    # initialise illumination spectrum for each subcell
    subcell_ill = set_subcell_spectrum(input_ill, subcell_eg, subcell_filter)

    subcell_jsc = [calc_jsc(subcell_ill[i], subcell_qe[i]) for i, _ in enumerate(subcell_qe)]

    print(subcell_jsc)
    iv_list = [gen_rec_iv(subcell_j01[i], subcell_j02[i], 1, 2, cell_temperature, 1e15, subcell_voltage, subcell_jsc[i]) \
               for i, _ in enumerate(subcell_eg)]

    if replace_iv != None:
        tmpvolt, tmpcurrent = replace_iv[1]
        tmpcurrent = tmpcurrent - subcell_jsc[replace_iv[0]]
        iv_list[replace_iv[0]] = (tmpvolt, tmpcurrent)

    # with series resistance, add the resistance to the first junction
    if rs > 0:
        iv_list[0] = gen_rec_iv_with_rs_by_newton(subcell_j01[0], subcell_j02[0], \
                                                  1, 2, cell_temperature, 1e15, rs, subcell_voltage, subcell_jsc[0])

    # plt.plot(iv_list[0][0],iv_list[0][1],'o')
    #plt.show()
    #plt.close()


    v, i = solve_mj_iv(iv_list, i_max=20)

    #plt.plot(v,i,'o')
    #plt.xlim([-1,10])
    #plt.show()

    #for iv in iv_list:
    #    plt.plot(iv[0], iv[1], '*', hold=True)
    #plt.plot(v, i, 'o-')
    #plt.ylim((-200, 0))
    #plt.show()
    #plt.savefig("result_iv.pdf")
    #plt.close()
    conv_efficiency = max_power(v, i) / input_ill.total_power()

    return conv_efficiency


if __name__ == "__main__":
    import yaml

    input_file = "db_layers.yaml"

    file_inst = open(input_file, 'r')

    file_content = file_inst.read()

    d = yaml.load(file_content)

    eta = calc_mj_eta(**d)

    print(eta)
