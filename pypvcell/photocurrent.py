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
from typing import Tuple, List
import numpy as np
import scipy.constants as sc
from pypvcell.illumination import Illumination
from pypvcell.spectrum import Spectrum, _energy_to_length


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


def gen_sub_qe_array(eg1, qe1, eg2, qe2,
                     qe_below_edge=1e-6, wl_bound=(0.0001, 5), ouput_type="spectrum"):
    edge_step = 1e-4
    lb = wl_bound[0]
    ub = wl_bound[1]

    qe = [[lb, qe_below_edge],
          [eg2 - edge_step, qe_below_edge],
          [eg2, qe2],
          [eg1 - edge_step, qe2],
          [eg1, qe1],
          [ub, qe1]]
    qe = np.array(qe)

    if ouput_type == "spectrum":
        qe = Spectrum(x_data=qe[:, 0], y_data=qe[:, 1], x_unit='eV')

    return qe


def lambert_abs(absorption: List[Spectrum], layer_thicknesses: List[float]):
    """
    Calculate transmission of stacked layers using Beer-Lambert's law

    :param absorption: an array absorption class, the unit of absorption should be 1/m
    :param layer_thicknesses: an array of layer thickness
    :return: transmission
    """

    assert len(absorption) == len(layer_thicknesses)

    standard_x, y = absorption[0].get_spectrum('m')

    term = np.zeros_like(y)
    for idx in range(len(absorption)):
        x, y = absorption[idx].get_interp_spectrum(standard_x, to_x_unit='m')
        term += y * layer_thicknesses[idx]

    t = np.exp(-term)
    return t


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

    ix, _ = input_illumination.get_spectrum(to_x_unit='m')
    qx, _ = qe.get_spectrum(to_x_unit='m')

    lower_bound = max([ix[0], qx[0]])
    upper_bound = min([ix[-1], qx[-1]])

    # Rearrange the new x-axis. This procedure ensures that the new wavelength range is the intersection of
    # illumination spectrum and qe
    new_x = np.concatenate((ix, qx))
    new_x = np.sort(new_x)
    new_x = new_x[(new_x > lower_bound) & (new_x < upper_bound)]
    new_x = np.concatenate(([lower_bound], new_x, [upper_bound]))

    ill_array = input_illumination.get_interp_spectrum(new_x, to_x_unit='m', to_y_area_unit='m**-2',
                                                       to_photon_flux=True,
                                                       interp_left=0, interp_right=0, raise_error=False)

    qe_array = qe.get_interp_spectrum(new_x, to_x_unit='m')

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

    bg_m = _energy_to_length(eg, 'eV', 'm')

    ill_array = input_illumination.get_interp_spectrum(input_illumination.core_x[input_illumination.core_x <= bg_m],
                                                       to_x_unit='m',
                                                       to_y_area_unit='m**-2', to_photon_flux=True)

    jsc = sc.e * np.trapz(ill_array[1, :], ill_array[0, :])

    return jsc


def eqe_to_iqe(eqe, reflectivity):
    """
    calculate internal quantum efficiency from external quantum efficiency

    :param eqe:
    :type Spectrum
    :param reflectivity:
    :type Spectrum
    :return: IQE
    :type Spectrum
    """

    return eqe / (1 - reflectivity)
