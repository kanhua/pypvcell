import unittest
from units_system import UnitsSystem

us = UnitsSystem()


class UnitsSystemTestCase(unittest.TestCase):
    def test_compare_dimension(self):
        self.assertTrue(us.compare_dimension('m-1', 'cm-1'))
        self.assertFalse(us.compare_dimension('s-1', 'cm-1'))

        with self.assertRaises(ValueError):
            us.compare_dimension('s m', 's-1')
            us.compare_dimension('', 's-1')

    def test_compare_wl_dim(self):
        self.assertTrue(us.compare_wavelength_dimension('J', 'nm'))
        self.assertTrue(us.compare_wavelength_dimension('J-1', 'eV-1'))

        self.assertFalse(us.compare_wavelength_dimension('J-1', 'eV'))

        with self.assertRaises(ValueError):
            us.compare_dimension('s m', 's-1')
            us.compare_dimension('', 's-1')


if __name__ == '__main__':
    unittest.main()
