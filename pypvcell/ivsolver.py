from typing import List, Tuple
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import newton_krylov
import scipy.constants as sc
# from spectrum_base import spectrum_base
from pypvcell.spectrum import Spectrum
import copy
from pypvcell.fom import max_power



def gen_rec_iv(j01, j02, n1, n2, temperature, rshunt, voltage, jsc=0):
    current = (j01 * (np.exp(sc.e * voltage / (n1 * sc.k * temperature)) - 1)
               + j02 * (np.exp(sc.e * voltage / (n2 * sc.k * temperature)) - 1) +
               voltage / rshunt) - jsc
    return (voltage, current)


def gen_rec_iv_by_rad_eta(j01,rad_eta,n1,temperature,rshunt,voltage,jsc=0):
    current = (j01/rad_eta * (np.exp(sc.e * voltage / (n1 * sc.k * temperature)) - 1)+
               voltage / rshunt) - jsc
    return (voltage, current)


def gen_rec_iv_with_rs_dont_work(j01, j02, n1, n2, temperature, rshunt, rseries, voltage, jsc=0):
    result = list()

    for v in voltage:
        init_guess = gen_rec_iv(j01, j02, n1, n2, temperature, rshunt, v, jsc)[1]

        def target_fun(current, voltage=0):
            sol = current - (j01 * (np.exp(sc.e * (v + rseries * current) / (n1 * sc.k * temperature)) - 1)
                             + j02 * (np.exp(sc.e * v / (n2 * sc.k * temperature)) - 1) +
                             v / rshunt) - jsc
            return sol

        tmp = newton_krylov(target_fun, init_guess)
        result.append(tmp)

    return (voltage, np.array(result))


def gen_rec_iv_with_rs_dont_work2(j01, j02, n1, n2, temperature, rshunt, rseries, voltage, jsc=0):
    """
    calculate the J-V with series resistance using naive iteration approach
    :param j01:
    :param j02:
    :param n1:
    :param n2:
    :param temperature:
    :param rshunt:
    :param rseries:
    :param voltage:
    :param jsc:
    :return:
    """

    current = (j01 * (np.exp(sc.e * voltage / (n1 * sc.k * temperature)) - 1) +
               voltage / rshunt) - jsc

    for i in range(0, 100, 1):
        next_current = j01 * (np.exp(sc.e * (voltage - rseries * current) / (n1 * sc.k * temperature)) - 1) + jsc
        residue = current - next_current
        current = copy.copy(next_current)

    return (voltage, current, residue)


def gen_rec_iv_with_rs_by_reverse(j01, j02, n1, n2, temperature, rshunt, rseries, voltage, jsc=0):
    voltage, current = gen_rec_iv(j01, j02, n1, n2, temperature, rshunt, voltage, jsc)

    current = np.linspace(-jsc * 1.1, 0, num=100)

    new_voltage = current * rseries + n1 * sc.k * temperature / sc.e * np.log((jsc + current) / j01)

    print(new_voltage)
    # get voc
    voc = np.interp(0, current, new_voltage)

    return (new_voltage, -current)


def find_root_newton(f, fp, x_init,verbose=True):
    max_iter = 10000
    tolerance = 1e-4
    step_lambda = 0.1

    current_x = x_init
    convergence = 1
    for i in range(0, max_iter):
        next_x = current_x - step_lambda * f(current_x) / fp(current_x)
        convergence = abs(next_x - current_x) / current_x
        # if convergence<tolerance:
        if np.isclose(current_x, next_x, atol=0, rtol=tolerance):
            if verbose:
                print("reach tolerance: %s" % convergence)
            return next_x
        current_x = next_x

    if verbose:
        print("reach maximum iteration: %s" % convergence)
    return next_x


def gen_rec_iv_with_rs_by_newton(j01, j02, n1, n2, temperature, rshunt, rseries, voltage, jsc=0,verbose=True):

    voltage, cur = gen_rec_iv(j01, j02, n1, n2, temperature, rshunt, voltage, jsc)

    solved_current = list()
    for i, v in enumerate(voltage):
        # since j02 could be an array, so we need to treat these cases
        if isinstance(j02, np.ndarray) and j02.shape[0] == voltage.shape[0]:
            t_j02 = j02[i]
        else:
            t_j02 = j02

        def f(current):
            result = current - (j01 * (np.exp(sc.e * (v - rseries * current) / (n1 * sc.k * temperature)) - 1)
                                + t_j02 * (np.exp(sc.e * (v - rseries * current) / (n2 * sc.k * temperature)) - 1) +
                                v / rshunt) + jsc
            return result

        def fp(current):
            result = 1 - (-rseries * j01 * (np.exp(sc.e * (v - rseries * current) / (n1 * sc.k * temperature)) - 1)
                          - rseries * t_j02 * (np.exp(sc.e * (v - rseries * current) / (n2 * sc.k * temperature)) - 1))
            return result

        if solved_current:
            init_x = solved_current[i - 1]
        else:
            init_x = cur[i]
        solved_current.append(find_root_newton(f, fp, init_x,verbose=verbose))

    return voltage, np.array(solved_current)


j01_lead_term=np.power(sc.e, 4) * 2 * sc.pi / (np.power(sc.c, 2) * np.power(sc.h, 3))


def calculate_j01_from_qe(qe, n_c=3.5, n_s=1, threshold=1e-3, step_in_ev=1e-5, lead_term=None, T=300):
    """
    Calculate j01 from absorptivity (QE).
    :param T:
    :param n_c: the refractive index of the material
    :param n_s: the refractive index of surroundings
    :param qe: QE or absorptivity. A spectrum_base class
    :param threshold: ignore the QE whose values are under the threshold
    :param step_in_ev: meshgrid size when doing numerical integration trapz()
    :return: j01
    """
    assert isinstance(qe, Spectrum)

    # lead_term = np.power(sc.e,4) * 2 * (n_c ** 2) / (np.power(sc.h, 3) * np.power(sc.c, 2) * 2 * np.power(sc.pi,2))

    if lead_term is None:
        # the additional sc.e^3 comes from the unit of E. We use the unit of eV to do the integration
        # of Planck's spectrum. Note that the term E^2*dE gives three q in total.
        #lead_term = np.power(sc.e, 4) * 2 * sc.pi * (n_c ** 2+n_s**2) / (np.power(sc.c, 2) * np.power(sc.h, 3))
        lead_term=j01_lead_term*(n_c ** 2+n_s**2)

    qe_a = qe.get_spectrum(to_x_unit='eV')

    qe_a = qe.get_interp_spectrum(np.arange(np.min(qe_a[0, :]), np.max(qe_a[0, :]), step=step_in_ev), 'eV')

    qe_a = qe_a[:, qe_a[1, :] > threshold]

    v_t = sc.k * T / sc.e

    j01 = lead_term * np.trapz(qe_a[1, :] * np.power(qe_a[0, :], 2) / (np.exp(qe_a[0, :] / v_t) - 1), qe_a[0, :])

    return j01


def calculate_bed(qe, T=300):
    
    assert isinstance(qe, Spectrum)

    qe_a = qe.get_spectrum(to_x_unit='eV')

    qe_a = qe.get_interp_spectrum(np.arange(np.min(qe_a[0, :]), np.max(qe_a[0, :]), step=1e-6), 'eV')

    v_t = sc.k * T / sc.e

    y = qe_a[1, :] * np.power(qe_a[0, :], 2) / (np.exp(qe_a[0, :] / v_t) - 1)

    x = qe_a[0, :]

    return x, y


def get_v_from_j(voltage, current, target_current):
    interp = interp1d(x=current, y=voltage)
    return interp(target_current)


def calculate_j01(eg_in_ev, temperature, n1, n_c=3.5,n_s=1):
    """
    Calculate J01 for analytical expression
    :param eg_in_ev:
    :param temperature:
    :param n1: ideality factor
    :param n_c: refractive index of the cell
    :param n_s: refractive index of the surroundings
    :return:
    """

    eg = eg_in_ev * sc.e
    Term1 = 2 * sc.pi * (n_c**2+n_s**2) * sc.e / (
        np.power(sc.h, 3) * np.power(sc.c, 2))
    Term2 = sc.k * temperature * np.exp(-eg / (n1 * sc.k * temperature))
    Term3 = (eg * eg) + (2 * eg * sc.k * temperature) + (2 * np.power(sc.k,2) * np.power(temperature,2))

    j01 = Term1 * Term2 * Term3
    return j01


def calculate_j02_from_voc(j01, jsc, voc, t, n2):
    # extracts j02 from an iv curve
    # verified as identical to idl
    term1 = jsc - (j01 * np.exp(sc.e * voc / (sc.k * t)))
    # print term1
    term2 = np.exp(sc.e * voc / (n2 * sc.k * t))
    # print term2

    j02 = term1 / term2
    return j02


def calculate_j02_from_rad_eff(j01, radiative_efficiency, voltage, temperature, n2):
    # verified as identical to idl, within numerical inaccuracy
    # no longer at voc - this is now dependent on bias voltage.

    term1 = j01 * np.exp(sc.e * voltage / (n2 * sc.k * temperature))
    term2 = -1 + (1 / radiative_efficiency)

    j02 = term1 * term2

    return j02


def solve_ms_mj_iv(v_i,ill_power):
    """
    Calculate the efficiency of mechanical stack solar cell

    :param v_i: a list of (voltage,current) tuple. Voltage and current are 1D np arrays. The current density is in W/m^2
    :param ill_power: the illumnation power in W/m^2
    :return: efficiency
    """

    max_p_list=[]
    for v, i in v_i:

        max_p_list.append(max_power(v,i))

    return np.sum(max_p_list)/ill_power


def solve_mj_iv(v_i, i_max=None):
    """

    :param v_i: a list of (voltage,current) tuple
    :type v_i: List[Tuple[np.ndarray,np.ndarray]]
    """

    # Select the minimum of range maximums

    # Select the maximum of range minimums


    # check the input
    i_range_max = 0
    i_range_min = 0
    for idx, iv_tup in enumerate(v_i):

        assert isinstance(iv_tup, tuple)

        voltage, current = iv_tup

        assert isinstance(voltage, np.ndarray)
        assert isinstance(current, np.ndarray)

        if idx == 0:
            i_range_max = np.max(current)
            i_range_min = np.min(current)

        else:
            if np.max(current) < i_range_max:
                i_range_max = np.max(current)
            if np.min(current) > i_range_min:
                i_range_min = np.min(current)

    current_range = np.linspace(i_range_min, i_range_max, num=10000)

    if i_max is not None:
        if i_max > np.max(current_range):
            raise ValueError("i_max is larger than the maximum of input I-Vs")
        elif i_max < np.min(current_range):
            raise ValueError("i_max is smaller than the maximum of input I-Vs")
        else:
            current_range = np.linspace(i_range_min, i_max, num=10000)

    voltage_sum = 0
    for iv_tup in v_i:
        voltage, current = iv_tup

        voltage_sum = voltage_sum + get_v_from_j(voltage, current, current_range)

    if np.any(np.isnan(voltage_sum)):
        voltage_sum = voltage_sum[~np.isnan(voltage_sum)]
        current_range = current_range[~np.isnan(voltage_sum)]

        voltage_sum = np.hstack(([-0.5, voltage_sum]))
        current_range = np.hstack(([current_range[0] * (1 - 0.0001)], current_range))

    return voltage_sum, current_range
