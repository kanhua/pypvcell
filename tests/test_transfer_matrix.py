import unittest
import numpy as np

from pypvcell.transfer_matrix_optics import TMLayers

class TMCase(unittest.TestCase):
    def test_something(self):
        layers = ['Air', 'SiO2_2', 'TiO2_2', 'GaAs_2']
        thicknesses = [0, 170, 135, 300]

        tm_layer = TMLayers(layers, thicknesses, wl_range=np.arange(400, 1099, 1))

        R, T = tm_layer.get_RT()

if __name__ == '__main__':
    unittest.main()
