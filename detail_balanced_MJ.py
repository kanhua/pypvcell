"""
detalied balance calculation of many junction devices
"""

import numpy as np
import copy
from illumination import qe_filter, illumination
from fom import voc
from ivsolver import calculate_j01, calculate_j02_from_rad_eff, \
    gen_rec_iv, gen_rec_iv_with_rs_by_newton, solve_mj_iv, \
    calculate_j01_from_qe, gen_rec_iv_by_rad_eta,solve_ms_mj_iv
from fom import max_power
from photocurrent import gen_square_qe, calc_jsc,calc_jsc_from_eg
import scipy.constants as sc
# from spectrum_base import spectrum_base
from spectrum import Spectrum


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


def rad_to_voc(rad_eta, qe, max_voltage=1.9,spectrum="AM1.5g"):
    """
    Calculate Voc from given radiative efficiency numerically

    :param rad_eta: radiative efficiency (in ratio)
    :param qe: quantum efficiency, a spectrum_base instance
    :param max_voltage: the maximum voltage of the dark-IV, the default value is 1.9. A safe way is set it to the value of the band gap
    :return: the calculated Voc
    """

    assert isinstance(qe, Spectrum)
    test_voltage = np.linspace(-0.5, max_voltage, num=300)

    j01_t = calculate_j01_from_qe(qe)

    # j02_t = calculate_j02_from_rad_eff(j01_t, rad_eta, test_voltage, 300, n2=2)

    v_top, i_top = gen_rec_iv_by_rad_eta(j01_t, rad_eta, 1, 300, 1e10, test_voltage)

    top_voc = extract_voc(v_top, i_top, qe,spectrum=spectrum)

    return top_voc


def rad_to_voc_fast(rad_eta, qe, spectrum="AM1.5g", T=300):
    """
    Calculate Voc from given radiative efficiency analytically

    :param rad_eta: radiative efficiency (in ratio)
    :param qe: quantum efficiency, a spectrum_base instance
    :param max_voltage: the maximum voltage of the dark-IV, the default value is 1.9.
    A safe way is set it to the value of the band gap
    :return: the calculated Voc
    """

    assert isinstance(qe, Spectrum)

    j01_t = calculate_j01_from_qe(qe, T=T)

    # j02_t = calculate_j02_from_rad_eff(j01_t, rad_eta, test_voltage, 300, n2=2)


    jsc = calc_jsc(input_illumination=illumination(spectrum), qe=qe)

    voc = np.log(rad_eta * jsc / j01_t) * (sc.k * T / sc.e)

    return voc




def extract_voc(voltage, current, qe, spectrum="AM1.5g"):
    """
    Calculate Voc from given dark I-V
    :param voltage: voltage array of dark I-V
    :param current: current array of dark I-V
    :param qe: quantum efficiency, a spectrum_base instance
    :param spectrum: can be "AM1.5g", "AM1.5d"
    :return: the calculated Voc
    """

    input_ill = illumination(x_data=1, y_data=1, x_unit=null)
    jsc = calc_jsc(input_ill, qe=qe)

    gen_current = current - jsc

    return voc(voltage, gen_current)


def calc_ere(qe, voc, T=300, ill=illumination("AM1.5g", y_data=1, x_unit=null), verbose=0):
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

    if verbose>0:
        print(jsc)

    jd = calculate_j01_from_qe(qe, lead_term=None)

    #ere = np.exp(sc.e * voc / (sc.k * T)) * jd / jsc/(3.5**2*2)

    ere = np.exp(sc.e * voc / (sc.k * T)) * jd / jsc

    return ere

def calc_1j_eta(eg,qe,r_eta,cell_temperature=300, n_c=3.5,n_s=1,
                concentration=1, spectrum="AM1.5g",
                j01_method="qe"):
    """
    Calculate the 1J efficiency from given band gap and qe values

    :param eg: The band gap of material
    :param qe: A single value. We assume flat, step-like QE.
    :param r_eta: Radiative efficiency
    :param cell_temperature: default to 300K
    :param n_c: the refractive index of the semiconductor material, default is 3.5
    :param n_s: the refractiv index of surrouding material, default is 1
    :param concentration: default value is 1
    :param spectrum: default value is "AM1.5g"
    :return: the calculated efficiency
    """
    volt = np.linspace(-0.5, eg, num=300)
    qe_spec=gen_square_qe(eg,qe)

    ill = illumination(x_data=concentration, y_data=concentration, x_unit=null)

    if j01_method=="qe":
        j01 = calculate_j01_from_qe(qe_spec, n_c=n_c, n_s=n_s)
        jsc = calc_jsc(ill, qe_spec)
    elif j01_method=="eg":
        j01=calculate_j01(eg,temperature=cell_temperature,n1=1,n_c=n_c,n_s=n_s)
        jsc=calc_jsc_from_eg(ill,eg)

    volt,current=gen_rec_iv_by_rad_eta(j01,r_eta,1,cell_temperature,1e15,voltage=volt,jsc=jsc)

    return max_power(volt,current)/ill.total_power()



def calc_mj_eta(subcell_eg, subcell_qe, subcell_rad_eff, cell_temperature, concentration=1, rs=0, replace_iv=None,
                replace_qe=None, verbose=0, spectrum="AM1.5g", n_s=1, mj="2T"):
    """

    :param subcell_eg:
    :param subcell_qe:
    :param subcell_rad_eff:
    :param cell_temperature:
    :param concentration:
    :param rs:
    :param replace_iv:
    :param replace_qe:
    :param verbose:
    :param spectrum:
    :param n_s:
    :param mj: "2T" for two terminal device. "MS" for multi-terminal mechanical stack.
    :return:
    """
    subcell_eg = np.array(subcell_eg)
    subcell_qe = np.array(subcell_qe)
    subcell_rad_eff = np.array(subcell_rad_eff)

    subcell_voltage = np.linspace(-0.5, 1.9, num=300)

    # calculate j01 and j02 for each subcell

    subcell_qe = [gen_square_qe(subcell_eg[i], subcell_qe[i]) for i, _ in enumerate(subcell_eg)]


    #subcell_j01 = [calculate_j01_from_qe(qe) for i,qe in enumerate(subcell_qe)]

    subcell_j01=[]
    for i, qe in enumerate(subcell_qe):
        if i==0:
            subcell_j01.append(calculate_j01_from_qe(qe, n_s=n_s))
        else:
            subcell_j01.append(calculate_j01_from_qe(qe, n_s=3.5))



    #subcell_j02 = [calculate_j02_from_rad_eff(subcell_j01[i], subcell_rad_eff[i], subcell_voltage, cell_temperature,
    #                                          2) for i, _ in enumerate(subcell_eg)]

    if replace_qe != None:
        subcell_qe[replace_qe[0]] = replace_qe[1]

    # calculate photocurrent for each subcell
    input_ill = illumination(x_data=concentration, y_data=concentration, x_unit=null)
    subcell_filter = [qe_filter(qe.core_wl, qe.core_spec, 'm') for qe in subcell_qe]

    # initialise illumination spectrum for each subcell
    subcell_ill = set_subcell_spectrum(input_ill, subcell_eg, subcell_filter)

    subcell_jsc = [calc_jsc(subcell_ill[i], subcell_qe[i]) for i, _ in enumerate(subcell_qe)]

    if verbose > 0:
        print(subcell_jsc)

    # iv_list = [gen_rec_iv(subcell_j01[i], subcell_j02[i], 1, 2, cell_temperature, 1e15, subcell_voltage, subcell_jsc[i]) \
    #           for i, _ in enumerate(subcell_eg)]

    iv_list = [gen_rec_iv_by_rad_eta(subcell_j01[i], subcell_rad_eff[i], 1, cell_temperature, 1e15, subcell_voltage,
                                     subcell_jsc[i]) \
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
    # plt.show()
    # plt.close()

    if mj=="2T":
        v, i = solve_mj_iv(iv_list, i_max=20)
        conv_efficiency = max_power(v, i) / input_ill.total_power()
    elif mj=="MS":
        conv_efficiency=solve_ms_mj_iv(iv_list,input_ill.total_power())

    # plt.plot(v,i,'o')
    # plt.xlim([-1,10])
    # plt.show()

    # for iv in iv_list:
    #    plt.plot(iv[0], iv[1], '*', hold=True)
    # plt.plot(v, i, 'o-')
    # plt.ylim((-200, 0))
    # plt.show()
    # plt.savefig("result_iv.pdf")
    # plt.close()


    return conv_efficiency


if __name__ == "__main__":
    import yaml

    input_file = "db_layers.yaml"

    file_inst = open(input_file, 'r')

    file_content = file_inst.read()

    d = yaml.load(file_content)

    eta = calc_mj_eta(**d)

    print(eta)
