from illumination import illumination, applyFilter, material_filter, BpFilter, qe_filter
import matplotlib.pyplot as plt
import numpy as np
from solcore3 import material, siUnits
from solcore3.plugins.adachi_alpha import create_adachi_alpha
from scipy.interpolate import interp1d
import scipy.constants as sc


def test_materialFilter():
    testIll = illumination()

    InGaP = material("GaInP")

    top_cell_material = InGaP(In=0.49, Na=siUnits(1e18, "cm-3"), role="p")

    energy_bounds = (0.29, 5, 1000)
    E_a, n, k, alpha_data = create_adachi_alpha("GaIn{}P".format(top_cell_material.In), Esteps=energy_bounds, T=300)
    top_cell_material.alphaE = interp1d(x=E_a / sc.e, y=alpha_data)

    bf = material_filter(testIll.wl_in_eV, top_cell_material, 1e-9)

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False)

    axs[0, 0].semilogy(testIll.wl_in_eV, top_cell_material.alphaE(testIll.wl_in_eV))

    axs[0, 1].plot(testIll.wl_in_eV, bf.filter_attenuation)

    plt.show()

def test_qe_filter():

    qe_array=gen_sqaure_QE(1.42, 0.9)
    test_ill = illumination()
    test_qef = qe_filter(test_ill.wl_in_eV, qe_array[:, 0], qe_array[:, 1])

    filtered_ill=applyFilter(test_ill,[test_qef])

    assert isinstance(filtered_ill,illumination)

    plt.plot(filtered_ill.wl_in_eV, filtered_ill.photon_flux_perEV)

    plt.show()



def gen_sqaure_QE(bandEdge_in_eV, QE_in_ratio):

    return np.array([[0,1e-3],[bandEdge_in_eV,1e-3],[bandEdge_in_eV+0.0001,QE_in_ratio],[5,QE_in_ratio]])



def test_applyFilter():
    testIll = illumination()

    InGaP = material("GaInP")

    top_cell_material = InGaP(In=0.49, Na=siUnits(1e18, "cm-3"), role="p")

    energy_bounds = (0.29, 5, 1000)
    E_a, n, k, alpha_data = create_adachi_alpha("GaIn{}P".format(top_cell_material.In), Esteps=energy_bounds, T=300)
    top_cell_material.alphaE = interp1d(x=E_a / sc.e, y=alpha_data)

    bf1 = material_filter(testIll.wl_in_eV, top_cell_material, 30e-9)
    bf2 = material_filter(testIll.wl_in_eV, top_cell_material, 120e-9)

    filteredIll = applyFilter(testIll, [bf1, bf2])

    plt.semilogy(testIll.wl_in_eV, testIll.photon_flux_in_W_perEV,
                 filteredIll.wl_in_eV, filteredIll.photon_flux_in_W_perEV)
    plt.show()


def test_total_power():
    testIll = illumination()
    print(testIll.total_power())


test_applyFilter()
#test_total_power()
#test_materialFilter()
#test_applyFilter()
#test_qe_filter()