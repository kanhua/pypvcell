import numpy as np
import scipy.interpolate
import scipy.constants as sc
from illumination import illumination
from spectrum import Spectrum


def gen_square_qe_array(bandEdge_in_eV, qe_in_ratio, qe_below_edge=1e-3, wl_bound=(0.01, 5)):
    """
    Generate a QE array :
    EQE(E)= qe if E >= Eg
    EQE(E)= z (z~0) if E<Eg
    :param bandEdge_in_eV: set Eg
    :param qe_in_ratio:  set qe
    :param qe_below_edge: set z
    :param wl_bound: tuple: (minE, maxE), The minimum and maximum of photon energy of this array
    :return: A 4x2 array, the first column is photon energy (in eV) and the second column is the qe
    """
    edge_step = 1e-4
    return np.array([[wl_bound[0], qe_below_edge], [bandEdge_in_eV, qe_below_edge],
                     [bandEdge_in_eV + edge_step, qe_in_ratio], [wl_bound[1], qe_in_ratio]])


def gen_square_qe(bandEdge_in_eV, qe_in_ratio, qe_below_edge=1e-6, wl_bound=(0.0001, 5)):
    """
    Generate a QE array. Same as gen_square_qe() except that it generates a Spectrum class object.
    EQE(E)= qe if E >= Eg
    EQE(E)= z (z~0) if E<Eg
    :param bandEdge_in_eV: set Eg
    :param qe_in_ratio:  set qe
    :param qe_below_edge: set z
    :param wl_bound: tuple: (minE, maxE), The minimum and maximum of photon energy of this array
    :return: A Spectrum object
    """
    qe_array = gen_square_qe_array(bandEdge_in_eV, qe_in_ratio,
                                   qe_below_edge=qe_below_edge, wl_bound=wl_bound)

    output_spec = Spectrum(x_data=qe_array[:, 0], y_data=qe_array[:, 1], x_unit="eV")

    return output_spec


def gen_qe_from_abs(absorption, layer_thickness):
    """
    Calculate the QE (absorptivity) from absorption coefficient and layer_thickness

    :param absorption: spectrum_base class, the unit of absorption: 1/m
    :param layer_thickness: layer thickness, unit: m
    :return: QE, a Spectrum class object
    """

    assert isinstance(absorption, Spectrum)

    wl, alpha = absorption.get_spectrum('m')

    abty = 1 - np.exp(-alpha * layer_thickness)

    qe = Spectrum(x_data=wl, y_data=abty, x_unit='m')

    return qe


def calc_jsc(input_illumination, qe, check_type=True):
    """
    Calculate Jsc from given QE

    :param check_type: boolean. Set whether check the type of input parameters. Set False for the concern of
    :param input_illumination: illumination spectrum, an spectrum_base object
    :param qe: QE, an spectrum_base object
    :return: value of Jsc (A/m^2)
    """

    if check_type == True:
        assert isinstance(input_illumination, illumination)
        assert isinstance(qe, Spectrum)

    # initialise a QE interp object

    ill_array = input_illumination.get_spectrum_density('m-2', 'eV', flux="photon")

    qe_array = qe.get_interp_spectrum(ill_array[0, :], 'eV')

    return sc.e * np.trapz(ill_array[1, :] * qe_array[1, :], ill_array[0, :])


def calc_jsc_from_eg(input_illumination, eg):
    """
    Calculate the Jsc by assuming 100% above-band-gap EQE

    :param input_illumination: illumination (class)
    :param eg: Band gap of the material (in eV)
    :return: value of Jsc (A/m^2)
    """

    assert isinstance(input_illumination, illumination)

    ill_array = input_illumination.get_spectrum_density('m-2', 'eV', flux="photon")

    ill_array = input_illumination.get_interp_spectrum_density(np.linspace(eg, ill_array[0, :].max(), num=100), "m-2",
                                                               "eV", flux="photon")

    jsc = sc.e * np.trapz(ill_array[1, :], ill_array[0, :])

    return jsc
