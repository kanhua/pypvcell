import numpy as np
import scipy.interpolate
import scipy.constants as sc
from illumination import illumination
from spectrum_base import spectrum_base

def gen_square_qe_array(bandEdge_in_eV, qe_in_ratio,qe_below_edge=1e-3,wl_bound=(0.01,5)):

    return np.array([[wl_bound[0],qe_below_edge],[bandEdge_in_eV,qe_below_edge],
                     [bandEdge_in_eV+0.0001,qe_in_ratio],[wl_bound[1],qe_in_ratio]])


def gen_square_qe(bandEdge_in_eV, qe_in_ratio,qe_below_edge=1e-4,wl_bound=(0.01,5)):

    qe_array=gen_square_qe_array(bandEdge_in_eV, qe_in_ratio,
                        qe_below_edge=qe_below_edge,wl_bound=wl_bound)

    output_spec=spectrum_base()
    output_spec.set_spectrum(qe_array[:,0],qe_array[:,1],'eV')

    return output_spec


def gen_qe_from_abs(absorption,layer_thickness):
    """
    Calculate the QE (absorptivity) from absorption coefficient and layer_thickness
    :param absorption: spectrum_base class, the unit of absorption: 1/m
    :param layer_thickness: layer thickness, unit: m
    :return: QE, as spectrum_base class
    """

    assert isinstance(absorption,spectrum_base)

    alpha_array=absorption.get_spectrum('m')

    abty=1-np.exp(-alpha_array[:,1]*layer_thickness)

    qe=spectrum_base()
    qe.set_spectrum(alpha_array[:,0],abty,'m')

    return qe



def calc_jsc(input_illumination,qe):

    assert isinstance(input_illumination,illumination)
    assert isinstance(qe,spectrum_base)

    # initialise a QE interp object

    ill_array=input_illumination.get_spectrum_density('m-2','eV',flux="photon")

    qe_array=qe.get_interp_spectrum(ill_array[:,0],'eV')

    return sc.e*np.trapz(ill_array[:,1]*qe_array[:,1],ill_array[:,0])