"""
This module collects the functions that solves I-V characteristics, including:

1. Generate the IV characteristics from known J01, J01 , n1, n2
2. Get J01 and J02 values from band gap or known EQEs using a detailed balance model
3. Add series resistance into a known I-V characterisitcs
4. Solve the I-V characteristics of multi-junction cell from the known I-Vs of each subcell


"""

"""
   Copyright 2017 Kan-Hua Lee, Toyota Technological Institute

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from typing import List, Tuple, Callable,Iterable
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import newton_krylov
import scipy.constants as sc
from .spectrum import Spectrum
import copy
from .fom import max_power


# from .solarcell import SolarCell


def gen_rec_iv(j01, j02, n1, n2, temperature, rshunt, voltage, jsc=0):
    current = (j01 * (np.exp(sc.e * voltage / (n1 * sc.k * temperature)) - 1)
               + j02 * (np.exp(sc.e * voltage / (n2 * sc.k * temperature)) - 1) +
               voltage / rshunt) - jsc
    return (voltage, current)


def gen_rec_iv_by_rad_eta(j01, rad_eta, n1, temperature,
                          rshunt, voltage, jsc=0, minus_one=True,plug_in_term=None):
    """
    Calculate recombination current from voltage

    :param j01:
    :param rad_eta:
    :param n1:
    :param temperature:
    :param rshunt:
    :param voltage:
    :param jsc:
    :param minus_one: True to add -1 in the exponential term
    :return:
    """
    if np.isinf(rshunt):
        shunt_term = 0.0
    else:
        shunt_term = voltage / rshunt

    m = sc.e / (n1 * sc.k * temperature)

    if minus_one:
        min_one = 1.0
    else:
        min_one = 0.0

    current = (j01 / rad_eta * (np.exp(m * voltage) - min_one) +
               shunt_term) - jsc
    if plug_in_term is not None:
        current+=plug_in_term(voltage)

    return voltage, current


def one_diode_v_from_i_p(current, j01, rad_eta, n1, temperature, jsc):
    m = rad_eta / j01
    return (n1 * sc.k * temperature / sc.e) * (m / (m * (current + jsc) + 1))


def one_diode_v_from_i(current, j01, rad_eta, n1, temperature, jsc, minus_one=True):
    """
    Calculate the voltage from demand current.
    This implementation drops the element that does not have log value, i.e. any log(x) that x<0


    :param current: demand current. This can be a numpy.nd array or a tuple (-Jsc, Jd)
    :type current: numpy.ndarray or tuple
    :param j01: saturation current density
    :type j01: float
    :param rad_eta: radiative efficiency
    :type rad_eta: float
    :param n1: diode factor
    :type n1: float
    :param temperature: temperature in K
    :type temperature: float
    :param jsc: Jsc. It has to be a positive value.
    :type jsc: float
    :param minus_one: set True to add one in exponential term
    :return: voltage, current, indexes that the values were kept
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """

    if jsc < 0:
        raise ValueError("Jsc should be a positive value")

    if isinstance(current, tuple):
        c1 = current[0] + jsc
        total_current = c1 + current[1]
    else:
        total_current = current + jsc

    log_component = rad_eta * (total_current) / j01

    if minus_one:
        log_component += 1.0

    if hasattr(log_component, "__iter__"):
        index = np.array(log_component) > 0
        n_current = total_current[index]
        log_component = log_component[index]
    else:
        n_current = 1
        index = 0
        if log_component < 0:
            return np.nan, n_current, index

    return (n1 * sc.k * temperature / sc.e) * np.log(log_component), n_current, index


def gen_rec_iv_with_rs_by_reverse(j01, j02, n1, n2, temperature, rshunt, rseries, voltage, jsc=0):
    voltage, current = gen_rec_iv(j01, j02, n1, n2, temperature, rshunt, voltage, jsc)

    current = np.linspace(-jsc * 1.1, 0, num=100)

    new_voltage = current * rseries + n1 * sc.k * temperature / sc.e * np.log((jsc + current) / j01)

    print(new_voltage)
    # get voc
    voc = np.interp(0, current, new_voltage)

    return (new_voltage, -current)


def find_root_newton(f, fp, x_init, verbose=True):
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


def gen_rec_iv_with_rs_by_newton(j01, j02, n1, n2, temperature, rshunt, rseries, voltage, jsc=0, verbose=True):
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
        solved_current.append(find_root_newton(f, fp, init_x, verbose=verbose))

    return voltage, np.array(solved_current)


j01_lead_term = np.power(sc.e, 4) * 2 * sc.pi / (np.power(sc.c, 2) * np.power(sc.h, 3))


def calculate_j01_from_qe(qe: Spectrum, n_c=3.5, n_s=1, threshold=1e-3, step_in_ev=1e-5, lead_term=None, T=300):
    r"""
    Calculate j01 from known absorptivity or QE using the following expression:
    
    .. math::
        J_{01}=\frac{2\pi q (n_c^2+n_s^2)}{\mbox{h}^3 \mbox{c}^2}\int_{0}^{\infty} \frac{a(E)E^2 dE}{\exp\left(\frac{E}{kT}\right)-1}
        
    
    :param T: temperature in Kelvin
    :type T: float
    :param n_c: the refractive index of the material
    :type n_c: float
    :param n_s: the refractive index of surroundings
    :type n_s: float
    :param qe: QE or absorptivity.
    :type qe: Spectrum
    :param threshold: ignore the QE whose values are under the threshold
    :type threshold: float
    :param step_in_ev: meshgrid size when doing numerical integration trapz()
    :type step_in_ev: float 
    :return: j01
    :rtype: float
    """

    assert isinstance(qe, Spectrum)

    # lead_term = np.power(sc.e,4) * 2 * (n_c ** 2) / (np.power(sc.h, 3) * np.power(sc.c, 2) * 2 * np.power(sc.pi,2))

    if lead_term is None:
        # the additional sc.e^3 comes from the unit of E. We use the unit of eV to do the integration
        # of Planck's spectrum. Note that the term E^2*dE gives three q in total.
        # lead_term = np.power(sc.e, 4) * 2 * sc.pi * (n_c ** 2+n_s**2) / (np.power(sc.c, 2) * np.power(sc.h, 3))
        lead_term = j01_lead_term * (n_c ** 2 + n_s ** 2)

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


def calculate_j01(eg_in_ev, temperature, n1, n_c=3.5, n_s=1, approx=False):
    r"""
    Calculate the saturation radiative recombination current J01 from known band gap using the following expression:
    
    .. math::
        J_{01}=\frac{2\pi q (n_c^2+n_s^2)}{\mbox{h}^3 \mbox{c}^2}\int_{0}^{E_g} \frac{E^2 dE}{\exp\left(\frac{E}{kT}\right)-1}
    
    
    If the parameter ``approx`` is set True, it uses an approximation of the above equation to calculate J01:
    
    
    .. math::
        J_{01}=\frac{2\pi k T q(n_c^2+n_s^2)}{\mbox{h}^3\mbox{c}^2}\exp(\frac{-E_g}{nkT})\left(E_g^2+2E_gkT+2k^2T^2\right)
    

    :param eg_in_ev: band gap in eV
    :type eg_in_ev: float
    :param temperature: temperature in K
    :type temperature: float
    :param n1: ideality factor
    :type n1: float
    :param n_c: refractive index of the cell
    :type n_c: float
    :param n_s: refractive index of the surroundings
    :type n_s: float
    :param approx: Set ``true`` to use approximation    
    :type approx: bool
    :return: the value of J01
    :rtype: float
    """

    eg = eg_in_ev * sc.e
    Term1 = 2 * sc.pi * (n_c ** 2 + n_s ** 2) * sc.e / (
            np.power(sc.h, 3) * np.power(sc.c, 2))
    Term2 = sc.k * temperature * np.exp(-eg / (n1 * sc.k * temperature))
    if approx == False:
        Term3 = np.power(eg, 2) + (2 * eg * sc.k * temperature) + (2 * np.power(sc.k, 2) * np.power(temperature, 2))
    else:
        Term3 = np.power(eg, 2)

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


def solve_ms_mj_iv(v_i, ill_power):
    """
    Calculate the efficiency of mechanical stack solar cell.
    It adds up the maximum power of each subcell and divde them by the illumination power.
    

    :param v_i: Voltage and current are 1D numpy arrays. The current density is in W/m^2
    :type v_i: tuple(numpy.ndarray, numpy.ndarray)
    :param ill_power: the illumnation power in W/m^2
    :type v_i: float
    :return: efficiency
    :rtype: float
    """

    max_p_list = []
    for v, i in v_i:
        max_p_list.append(max_power(v, i))

    return np.sum(max_p_list) / ill_power


def solve_mj_iv(v_i, i_max=None, discret_num=10000):
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

    current_range = np.linspace(i_range_min, i_range_max, num=discret_num)

    if i_max is not None:
        if i_max > np.max(current_range):
            raise ValueError("i_max is larger than the maximum of input I-Vs")
        elif i_max < np.min(current_range):
            raise ValueError("i_max is smaller than the maximum of input I-Vs")
        else:
            current_range = np.linspace(i_range_min, i_max, num=discret_num)

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


def new_solve_mj_iv(v_i, i_max=None, disc_num=1000, verbose=0):
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

    v, i = solve_iv_range(v_i, i_range_min, i_range_max, disc_num=disc_num)

    p = v * (-i)

    nv = v
    ni = i
    for idx in range(8):
        i_index = np.argmax(p)

        if verbose > 0:
            print("Jmp :%s" % ni[i_index])
            print("Vmp :%s" % nv[i_index])
            print("max power:%s" % p[i_index])
        nv, ni = solve_iv_range(v_i, ni[max(i_index - 5, 0)],
                                ni[min(i_index + 5, len(ni) - 1)], disc_num=disc_num)

        p = nv * (-ni)

    voltage_sum = np.concatenate((np.array([-0.5]), v, nv))

    current_range = np.concatenate((np.array([i[0] * (1 - 0.0001)]), i, ni))

    ag = np.argsort(voltage_sum)

    return voltage_sum[ag], current_range[ag]


def solve_mj_iv_obj_with_optimization(subcells, i_max=None, disc_num=1000, verbose=0):
    """
    Solve the I-V of MJ cell from given subcells, namely
    
    .. math::
        V_{tot}(I_m)=\sum_{i=1}^N V_i(I_{m})
        
    This function automatically choose the appropriate values of current

    :param subcells: a list of SolarCell objects
    :param i_max: the maximum
    :param disc_num: number of discretization
    :type disc_num: int
    :param verbose: display the intermediate results
    :type verbose: int
    :return: solved (voltage, current)
    :rtype: tuple(numpy.ndarray, numpy.ndarray) 
    """

    # Determine the range of current
    for idx, sc in enumerate(subcells):

        voltage, current = sc.get_iv()

        if idx == 0:
            i_range_max = np.max(current)
            i_range_min = np.min(current)

        else:
            if np.max(current) < i_range_max:
                i_range_max = np.max(current)
            if np.min(current) > i_range_min:
                i_range_min = np.min(current)

    # Solve the first iteration
    v, i = solve_iv_range_obj(subcells, i_range_min, i_range_max, disc_num=disc_num)

    p = v * (-i)
    if np.max(p) < 0:
        raise ValueError("Negative maximum power value")

    # Solve the next few iterations
    nv = v
    ni = i

    voltage_sum = v
    current_range = i
    for idx in range(8):
        i_index = np.argmax(p)

        if verbose > 0:
            print("Jmp :%s" % ni[i_index])
            print("Vmp :%s" % nv[i_index])
            print("max power:%s" % p[i_index])
        nv, ni = solve_iv_range_obj(subcells, ni[max(i_index - 20, 0)],
                                    ni[min(i_index + 20, len(ni) - 1)], disc_num=disc_num)

        voltage_sum = np.concatenate((voltage_sum, nv))
        current_range = np.concatenate((current_range, ni))

        p = nv * (-ni)

    # Extrapolate the short circuit current.
    voltage_sum = np.concatenate((np.array([-0.5]), voltage_sum))

    current_range = np.concatenate((np.array([i[0] * (1 - 0.0001)]), current_range))

    # Sort the final result
    ag = np.argsort(voltage_sum)

    return voltage_sum[ag], current_range[ag]


def solve_iv_range(v_i, i_min, i_max, disc_num=1000):
    """
    Calculate the voltages of a series-connected solar cells I-Vs for each current point Im:

    .. math::
        V_{tot}(I_m)=\sum_{i=1}^N V_i(I_{m})
            
    :param v_i: a list of (voltage, current) tuple of subcells
    :type v_i: list[tuple(numpy.ndarray,numpy.ndarray)]
    :param i_min: The minimum of the range of the current
    :type i_min: float
    :param i_max: The maximum of the range of the current
    :type i_max: float
    :param disc_num: The number of points of current to be discretized
    :type disc_num: float
    :return: voltage, current
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """

    current_range = np.linspace(i_min, i_max, num=disc_num)
    voltage_sum = 0
    for iv_tup in v_i:
        voltage, current = iv_tup

        voltage_sum = voltage_sum + get_v_from_j(voltage, current, current_range)

    if np.any(np.isnan(voltage_sum)):
        voltage_sum = voltage_sum[~np.isnan(voltage_sum)]
        current_range = current_range[~np.isnan(voltage_sum)]

    return voltage_sum, current_range


def solve_iv_range_obj(subcells: List, i_min, i_max, disc_num=1000):
    """
    Calculate the voltages of a MJ cell from a given range of J and subcells.
    It finds a new V(J) such that V(J)=sum(V_i(J_i)), where V_i(J_i) is
    the J-V characteristics of the subcell i.
    This function only does one-run without iterating. The algorithm is the same as
    ``solve_iv_range()`` but this function takes ``SolarCell`` objects as input rather
    than a list of (V,J) tuples.

    :param subcells: A list of SolarCell objects
    :type subcells: list[solarcell.SolarCell]
    :param i_min: The minimum of the range of the current 
    :type i_min: float
    :param i_max: The maximum of the range of the current
    :type i_max: float
    :param disc_num: number of descretions between i_min and i_max
    :type disc_num: float
    :return: voltage, current
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """

    # discretize positive and negative current separately
    n_i_range = np.linspace(i_min, 0, num=int(disc_num / 2))
    p_i_range = np.linspace(0, i_max, num=int(disc_num / 2))

    # rearrange the discretized points
    current_range = np.sort(np.unique(np.concatenate((n_i_range, p_i_range))))

    # set up an empty array for the range
    voltage_sum = np.zeros_like(current_range)

    for sc in subcells:
        volt, current, index = sc.get_v_from_j(current_range)

        voltage_sum = voltage_sum[index]
        current_range = current_range[index]

        voltage_sum = voltage_sum + volt

    # Drop any Nan values
    if np.any(np.isnan(voltage_sum)):
        voltage_sum = voltage_sum[~np.isnan(voltage_sum)]
        current_range = current_range[~np.isnan(voltage_sum)]

    return voltage_sum, current_range

def solve_v_from_j_adding_epsilon(iv_func:Callable[[np.ndarray],np.ndarray],
                                  current:np.ndarray,equation_solver_func,epsilon):
    """

    :param iv_func: a I(V) function
    :param current: an array of current that needs to be solved
    :param equation_solver_func: bisect, newton, or other scipy solvers
    :param epsilon: small offset added to current. The solver will solve current*(1+epsilon) and current*(1-epsilon)
    :return: a 2xN [voltage,current] array
    """

    def eqn_func(x,j0):
        return iv_func(x)-j0

    solved_iv_pair=[]
    if epsilon!=0:
        ratio=[1-epsilon,1+epsilon]
    else:
        ratio=[1]
    for j1 in current:
        for r in ratio:
            jj = j1 * r
            try:
                solved_iv_pair.append((equation_solver_func(eqn_func, -23, 5, args=jj), jj))
            except ValueError:
                print("no solution found for {}".format(jj))


    return np.array(solved_iv_pair)

def _clean_j_to_solve(j_to_solve:np.ndarray)->np.ndarray:

    nj=np.ravel(j_to_solve)

    nj=np.unique(nj)

    return nj

def _add_epsilon(j_to_solve:np.ndarray,epsilon:float)->np.ndarray:

    j_to_solve_minus=j_to_solve*(1-epsilon)
    j_to_solve_plus=j_to_solve*(1+epsilon)

    nj=np.concatenate((j_to_solve_minus,j_to_solve_plus))

    return np.sort(nj)




def solve_series_connected_ivs(iv_funcs:Iterable[Callable[[np.ndarray],np.ndarray]],
                               vmin:float,vmax:float,vnum:int=20):

    from scipy.optimize import bisect
    junc_num=len(iv_funcs)
    j_to_solve=np.empty((junc_num,vnum))
    volt=np.linspace(vmin,vmax,vnum)

    # fill in the current needs to be solved
    for idx,iv in enumerate(iv_funcs):
        j_to_solve[idx,:]=iv(volt)

    j_to_solve=_clean_j_to_solve(j_to_solve)

    j_to_solve=_add_epsilon(j_to_solve,epsilon=0.01)

    solved_v=np.empty((junc_num,j_to_solve.shape[0]))

    # solve the voltage from each current value
    for v_idx, iv in enumerate(iv_funcs):
        iv_values=solve_v_from_j_adding_epsilon(iv,j_to_solve,bisect,epsilon=0)
        solved_v[v_idx,:]=iv_values[:,0]

    iv_pair=np.empty((j_to_solve.shape[0],2))

    # summing up the voltage to get the series-connected voltage values
    iv_pair[:,0]=np.sum(solved_v,axis=0)
    iv_pair[:,1]=j_to_solve

    return iv_pair

