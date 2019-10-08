import unittest
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

from pypvcell.ivsolver import solve_series_connected_ivs
from pypvcell.illumination import load_astm
from pypvcell.solarcell import SQCell



def rev_diode(voltage):
    rev_j01 = 4.46e-15
    rev_bd_v=0.1
    return -rev_j01*np.exp(sc.e*(-voltage-rev_bd_v)/(sc.k*300)-1)

class SeriesConnectedCellTestCase(unittest.TestCase):
    def test_solving_five_cells(self):
        sq1_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=rev_diode)
        sq1_cell.set_input_spectrum(load_astm("AM1.5d") * 0.5)

        sq2_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=rev_diode)
        sq2_cell.set_input_spectrum(load_astm("AM1.5d"))

        sq3_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=rev_diode)
        sq3_cell.set_input_spectrum(load_astm("AM1.5d"))

        sq4_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=rev_diode)
        sq4_cell.set_input_spectrum(load_astm("AM1.5d") * 1.5)

        sq5_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=rev_diode)
        sq5_cell.set_input_spectrum(load_astm("AM1.5d") * 0.75)

        iv_funcs = [sq1_cell.get_j_from_v, sq2_cell.get_j_from_v, sq3_cell.get_j_from_v, sq4_cell.get_j_from_v,
                    sq5_cell.get_j_from_v]

        iv_pair = solve_series_connected_ivs(iv_funcs=iv_funcs, vmin=-3, vmax=3, vnum=300)

        volt = np.linspace(-3, 3, num=300)
        plt.plot(volt, sq1_cell.get_j_from_v(volt))
        plt.plot(volt, sq2_cell.get_j_from_v(volt))
        plt.plot(volt, sq4_cell.get_j_from_v(volt))
        plt.plot(volt, sq5_cell.get_j_from_v(volt))

        plt.plot(iv_pair[:, 0], iv_pair[:, 1], '.-')
        plt.ylim([-500, 0])
        plt.xlim([-5, 6])

        plt.xlabel("voltage (V)")
        plt.ylabel("currnet (A/m^2)")
        plt.show()




if __name__ == '__main__':
    unittest.main()
