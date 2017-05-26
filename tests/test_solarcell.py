import unittest
import numpy as np
import matplotlib.pyplot as plt

from pypvcell.illumination import load_astm
from pypvcell.solarcell import SQCell, MJCell, DBCell
from pypvcell.photocurrent import gen_step_qe


class SolarCellTestCase(unittest.TestCase):
    def setUp(self):
        self.input_ill = load_astm("AM1.5g")

    def test_sqcell_transmission(self):

        sq_cell = SQCell(eg=1.42, cell_T=293, n_c=1, n_s=1)
        sq_cell.set_input_spectrum(input_spectrum=self.input_ill)

        sp = sq_cell.get_transmit_spectrum()

        sp_arr = sp.get_spectrum(to_x_unit='eV')

        for i in range(sp_arr.shape[1]):
            if sp_arr[0, i] > 1.42:
                self.assertEqual(sp_arr[1, i], 0)

        if False:
            plt.plot(sp_arr[0, :], sp_arr[1, :])
            plt.show()

    def test_mjcell(self):

        sq_ingap = SQCell(eg=1.9, cell_T=293)
        sq_gaas = SQCell(eg=1.42, cell_T=293)

        dj_cell = MJCell([sq_ingap, sq_gaas])
        dj_cell.set_input_spectrum(self.input_ill)

        print("2J eta:%s" % dj_cell.get_eta())

    def test_3jcell(self):

        sq_ingap = SQCell(eg=1.9, cell_T=293)
        sq_gaas = SQCell(eg=1.42, cell_T=293)
        sq_ge = SQCell(eg=0.67, cell_T=293)

        tj_cell = MJCell([sq_ingap, sq_gaas, sq_ge])
        tj_cell.set_input_spectrum(self.input_ill)

        print("3J eta: %s" % tj_cell.get_eta())

    def test_dbcell(self):

        gaas_qe = gen_step_qe(1.42, 1)

        gaas_db = DBCell(qe=gaas_qe, rad_eta=1, T=300)

        gaas_db.set_input_spectrum(load_astm("AM1.5g"))

        gaas_db_eta = gaas_db.get_eta()

        sq_gaas = SQCell(eg=1.42, cell_T=300)
        sq_gaas.set_input_spectrum(load_astm("AM1.5g"))

        sq_gaas_eta = sq_gaas.get_eta()

        self.assertTrue(np.isclose(gaas_db_eta, sq_gaas_eta, rtol=5e-3))

    def test_dbcell2(self):

        qe_val = 0.8

        ill = load_astm("AM1.5g")

        gaas_qe = gen_step_qe(1.42, qe_val)

        gaas_db = DBCell(qe=gaas_qe, rad_eta=1, T=300)

        gaas_db.set_input_spectrum(ill)

        trans_sp = gaas_db.get_transmit_spectrum()

        # Compare the expected and transmitted spectrum

        x = np.linspace(1.5, 2, num=10)

        trans = trans_sp.get_interp_spectrum(x, to_x_unit='eV')

        orig = ill.get_interp_spectrum(x, to_x_unit='eV') * (1 - qe_val)

        cp = np.allclose(trans[1, :], orig[1, :])

        self.assertTrue(cp)


if __name__ == '__main__':
    unittest.main()
