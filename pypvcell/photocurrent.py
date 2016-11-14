from typing import Tuple
import numpy as np
import scipy.constants as sc
from pypvcell.illumination import Illumination
from pypvcell.spectrum import Spectrum


def gen_step_qe_array(bandEdge_in_eV, qe_in_ratio, qe_below_edge=1e-3, wl_bound=(0.01, 5)):
    """
    Generate a staircase QE array :
    EQE(E)= qe if E >= Eg
    EQE(E)= z (z~0) if E<Eg

    :param bandEdge_in_eV: set Eg
    :type bandEdge_in_eV: float
    :param qe_in_ratio:  set qe value (qe) above Eg
    :type qe_in_ratio: float
    :param qe_below_edge: set qe value (z) below Eg
    :param wl_bound: tuple: (minE, maxE), The minimum and maximum of photon energy of this array
    :type wl_bound: Tuple[float,float]
    :return: A 4x2 array, the first column is photon energy (in eV) and the second column is the qe
    :rtype: np.ndarray
    """
    edge_step = 1e-4
    return np.array([[wl_bound[0], qe_below_edge], [bandEdge_in_eV, qe_below_edge],
                     [bandEdge_in_eV + edge_step, qe_in_ratio], [wl_bound[1], qe_in_ratio]])


def gen_step_qe(bandEdge_in_eV, qe_in_ratio, qe_below_edge=1e-6, wl_bound=(0.0001, 5)):
    """
    Generate a staircase QE array. Same as gen_square_qe_array() except that it generates a Spectrum class object.
    EQE(E)= qe if E >= Eg
    EQE(E)= z (z~0) if E<Eg

    :param bandEdge_in_eV: set Eg
    :type bandEdge_in_eV: float
    :param qe_in_ratio:  set qe value (qe) above Eg
    :type qe_in_ratio: float
    :param qe_below_edge: set qe value (z) below Eg
    :param wl_bound: tuple: (minE, maxE), The minimum and maximum of photon energy of this array
    :type wl_bound: Tuple[float,float]
    :return: A Spectrum object
    :rtype: Spectrum
    """
    qe_array = gen_step_qe_array(bandEdge_in_eV, qe_in_ratio, qe_below_edge=qe_below_edge, wl_bound=wl_bound)

    output_spec = Spectrum(x_data=qe_array[:, 0], y_data=qe_array[:, 1], x_unit="eV")

    return output_spec


def conv_abs_to_qe(absorption, layer_thickness):
    """
    Calculate the QE (absorptivity) from absorption coefficient and layer_thickness
    Note that the unit of the absorption should match the layer thickness.
    For example, if the unit of ``absorption`` is 1/meter, the layer_thickness should be meter.

    :param absorption: Spectrum class instance, the unit of absorption: 1/m
    :type absorption: Spectrum
    :param layer_thickness: layer thickness, unit: m
    :type layer_thickness: float
    :return: QE, a Spectrum class instance
    :rtype: Spectrum
    """

    if isinstance(absorption, Spectrum) == False:
        raise TypeError("The parameter absorption should be a Spectrum class")

    wl, alpha = absorption.get_spectrum('m')

    abty = 1 - np.exp(-alpha * layer_thickness)

    qe = Spectrum(x_data=wl, y_data=abty, x_unit='m')

    return qe


def calc_jsc(input_illumination, qe):
    """
    Calculate Jsc from given QE and illumination

    :param input_illumination: illumination spectrum
    :type input_illumination: Illumination
    :param qe: QE
    :type qe: Spectrum
    :return: value of Jsc (A/m^2)
    """

    if not isinstance(input_illumination, Spectrum):
        raise TypeError("input_illumination should be a subclass of Spectrum")

    if not isinstance(qe, Spectrum):
        raise TypeError("qe should be an instance of Spectrum class")

    # initialise a QE interp object

    ix=input_illumination.core_x
    qx=qe.core_x

    new_x=np.concatenate((ix,qx))
    new_x=np.sort(new_x)

    ill_array=input_illumination.get_interp_spectrum(new_x,to_x_unit='m',to_y_area_unit='m**-2',to_photon_flux=True,
                                                     interp_left=0,interp_right=0,raise_error=False)

    qe_array=qe.get_interp_spectrum(new_x,to_x_unit='m')


    return sc.e * np.trapz(ill_array[1, :] * qe_array[1, :], ill_array[0, :])


def calc_jsc_from_eg(input_illumination, eg):
    """
    Calculate the Jsc by assuming 100% above-band-gap EQE

    :param input_illumination: illumination (class)
    :type input_illumination: Illumination
    :param eg: Band gap of the material (in eV)
    :return: value of Jsc (A/m^2)
    """

    if not isinstance(input_illumination, Spectrum):
        raise TypeError("input_illumination should be a subclass of Spectrum, preferably Illumination class")

    ill_array = input_illumination.get_spectrum(to_x_unit='eV', to_y_area_unit='m**-2', to_photon_flux=True)

    ill_array = input_illumination.get_interp_spectrum(np.linspace(eg, ill_array[0, :].max(), num=100), to_x_unit='eV',
                                                       to_y_area_unit='m**-2', to_photon_flux=True)

    jsc = sc.e * np.trapz(ill_array[1, :], ill_array[0, :])

    return jsc

def eqe_to_iqe(eqe,reflectivity):
    """
    calculate internal quantum efficiency from external quantum efficiency

    :param eqe:
    :type Spectrum
    :param reflectivity:
    :type Spectrum
    :return: IQE
    :type Spectrum
    """

    return eqe/(1-reflectivity)