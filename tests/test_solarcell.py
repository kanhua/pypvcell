import unittest
from pypvcell.illumination import Illumination
from pypvcell.solarcell import SQCell
import matplotlib.pyplot as plt


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.input_ill = Illumination("AM1.5g")

    def test_sqcell_transmission(self):

        sq_cell = SQCell(eg=1.42, cell_T=293, n_c=1, n_s=1)
        sq_cell.set_input_spectrum(input_spectrum=self.input_ill)

        sp = sq_cell.get_transmit_spectrum()

        sp_arr = sp.get_spectrum(to_x_unit='eV')

        for i in range(sp_arr.shape[1]):
            if sp_arr[0, i] > 1.42:
                self.assertEqual(sp_arr[1, i], 0)

        if True:
            plt.plot(sp_arr[0, :], sp_arr[1, :])
            plt.show()


if __name__ == '__main__':
    unittest.main()
