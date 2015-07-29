"""unit test for figure of merit package"""

__author__ = 'kanhua'

import unittest
import numpy as np
from fom import ff, voc, isc


class FoMTestCase(unittest.TestCase):
    def setUp(self):
        self.test_voltage = np.array([-1, 0.5, 2])
        self.test_current = np.array([-2, -0.5, 1])

    def test_voc(self):
        self.assertEqual(voc(self.test_voltage, self.test_current), 1)

    def test_isc(self):
        self.assertEqual(isc(self.test_voltage, self.test_current), -1)

    def test_ff(self):
        self.assertEqual(ff(self.test_voltage,
                            self.test_current), 0.25)


if __name__ == '__main__':
    unittest.main()
