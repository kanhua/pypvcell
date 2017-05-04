"""
This module calculates transfer matrix optics.
Basic usages look like this:

::

    # Setup the layers
    layers = ['Air', 'SiO2_2', 'TiO2_2', 'GaAs_2']
    
    # Assign the thickness to each layer. 
    # Assuimg the first and the last layers are infinite.
    thicknesses = [0, 170, 135, 300]

    tm_layer = TMLayers(layers, thicknesses, 
    wl_range=np.arange(400, 1099, 1))

    R, T = tm_layer.get_RT()

    plt.plot(tm_layer.wl_range, R, label='R')
    plt.plot(tm_layer.wl_range, T, label='T')
    

References:

Leif A. A. Pettersson, Lucimara S. Roman, and Olle Inganäs, Modeling photocurrent action spectra of photovoltaic devices based on organic thin
films, Journal of Applied Physics, vol. 86, pp. 487 (1999). DOI: `10.1063/1.370757 <http://dx.doi.org/10.1063/1.370757>_`

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

from os.path import join, isfile
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import warnings

MAT_DIR = './matdata'
MAT_PREFIX = 'nk_'


def get_ntotal(mat_name, lambdas):
    fname = join(MAT_DIR, "%s%s.csv" % (MAT_PREFIX, mat_name))
    fdata = np.loadtxt(fname, delimiter=',', skiprows=2, comments='#')

    wl = fdata[:, 0]
    nr = fdata[:, 1]
    nk = fdata[:, 2]

    int_nr = interp1d(wl, nr)
    int_nk = interp1d(wl, nk)

    ip_nr = int_nr(lambdas)
    ip_nk = int_nk(lambdas)

    nt = ip_nr + 1j * ip_nk

    return nt


def inter_mat(n1, n2):
    # transfer matrix at an interface
    r = (n1 - n2) / (n1 + n2)
    t = (2 * n1) / (n1 + n2)
    ones = np.ones_like(r)

    ret = np.array([[ones, r], [r, ones]], dtype=np.complex)
    ret = ret / t
    return ret


def prop_mat(layer_nr, layer_d, wavelength):
    # propagation matrix
    # n = complex dielectric constant
    # d = thickness
    # l = wavelength

    assert len(layer_nr) == len(wavelength)
    xi = (2 * np.pi * layer_d * layer_nr) / wavelength
    zeros = np.zeros_like(xi, dtype=np.complex)

    L = np.array([[np.exp(-1j * xi), zeros], [zeros, np.exp(1j * xi)]])
    return L


class TMLayers(object):
    def __init__(self, layers:List, thicknesses:List, wl_range=None):
        """
        Initialize the multi-layer system
        
        :param layers: list. The name of the layers. The layer name should be in the folder ``matdata/``. The first 
        :param thicknesses: The thickness of the layers.
        :param wl_range: ndarray. The reference wavelength range.
        """

        if wl_range is None:
            wl_range = np.arange(350, 999, 1)
        self.layers = layers
        self.thicknesses = thicknesses
        self.wl_range = wl_range
        self.nr = self._get_nr()
        self.s_mat = self._setup_S()

    def set_thickness(self, thicknesses: list):

        assert len(thicknesses) == len(self.layers)

        self.thicknesses = thicknesses

        self.s_mat = self._setup_S()

    def _get_nr(self):
        """
        Setup refractive index
        :return: 
        """
        n = np.zeros((len(self.layers), len(self.wl_range)), dtype=complex)
        for i, l in enumerate(self.layers):
            ni = np.array(get_ntotal(l, self.wl_range))
            n[i, :] = ni

        return n

    def _setup_I(self, nr):
        """
        Set up interface matrices I
        :return: 
        """
        i_mat = []
        for m in range(len(self.thicknesses) - 1):
            i_mat.append(inter_mat(nr[m, :], nr[m + 1, :]))

        return np.stack(i_mat)

    def _setup_L(self, nr):
        """
        Set up propogation matrices L
        :return: 
        """
        l_mat = []
        for m in range(1, len(self.thicknesses) - 1):
            l_mat.append(prop_mat(nr[m, :], self.thicknesses[m], self.wl_range))

        return np.stack(l_mat)

    def _setup_S(self):

        i_mat = self._setup_I(self.nr)
        l_mat = self._setup_L(self.nr)

        S = i_mat[0]

        for m in range(1, len(self.thicknesses) - 1):
            assert S.shape == l_mat[m - 1].shape
            assert S.shape == i_mat[m].shape
            S = np.einsum('ij...,jk...->ik...', S, l_mat[m - 1])
            S = np.einsum('ij...,jk...->ik...', S, i_mat[m])

        return S

    def get_RT_fast(self):
        """
        Calculate the reflection and transmission
        
        :return: Tuple: (Reflection array, Transmission array)
        """

        warnings.warn("Illumination class will be deprecated in future version.", DeprecationWarning)
        # initialize an array

        nr = self.nr
        S = self.s_mat

        R = np.abs(S[1, 0, :] / S[0, 0, :]) ** 2
        # T[ind] = abs((2 / (1 + n[0, ind]))) / np.sqrt(1 - R_glass[ind] * R[ind])
        T = np.power(np.abs(1 / S[0, 0, :]), 2) * nr[-1, :] / nr[0, :]

        return R, T

    def get_RT(self):

        # initialize an array

        nr = self.nr
        S = self.s_mat

        R = np.abs(S[1, 0, :] / S[0, 0, :]) ** 2
        # T[ind] = abs((2 / (1 + n[0, ind]))) / np.sqrt(1 - R_glass[ind] * R[ind])
        T = np.power(np.abs(1 / S[0, 0, :]), 2) * nr[-1, :] / nr[0, :]

        return R, T

    def _get_inv_S(self, S):
        """
        Calculate the inverse of S matrix
        
        :param S: 
        :return: 
        """


        invS = np.moveaxis(S, -1, 0)

        invS = np.linalg.inv(invS)

        invS = np.moveaxis(invS, 0, -1)

        return invS


if __name__ == "__main__":
    layers = ['Air', 'SiO2_2', 'TiO2_2', 'GaAs_2']
    thicknesses = [0, 170, 135, 300]

    tm_layer = TMLayers(layers, thicknesses, wl_range=np.arange(400, 1099, 1))

    R, T = tm_layer.get_RT()

    plt.plot(tm_layer.wl_range, R, label='R')
    plt.plot(tm_layer.wl_range, T, label='T')
    plt.plot(tm_layer.wl_range, R + T, label='R+T')
    plt.plot(tm_layer.wl_range, 1 - R - T, label='1-R-T')
    plt.legend()
    plt.savefig("demo_RT.png")
    plt.close()

    tm_layer = TMLayers(layers[::-1], thicknesses[::-1], wl_range=np.arange(400, 1099, 1))

    R, T = tm_layer.get_RT()

    plt.plot(tm_layer.wl_range, R, label='R')
    plt.plot(tm_layer.wl_range, T, label='T')
    plt.plot(tm_layer.wl_range, R + T, label='R+T')
    plt.plot(tm_layer.wl_range, 1 - R - T, label='1-R-T')
    plt.legend()
    plt.savefig("demo_RT_GaAs.png")
