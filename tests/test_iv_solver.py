import unittest
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import copy

from pypvcell.ivsolver import solve_series_connected_ivs
from pypvcell.illumination import load_astm
from pypvcell.solarcell import SQCell, MJCell


def rev_diode(voltage):
    rev_j01 = 4.46e-15
    rev_bd_v = 0.1
    return -rev_j01 * np.exp(sc.e * (-voltage - rev_bd_v) / (sc.k * 300) - 1)


def rev_breakdown_diode(voltage):
    rev_j01 = 4.46e-17
    rev_bd_v = 6
    return -rev_j01 * np.exp(sc.e * (-voltage - rev_bd_v) / (sc.k * 300) - 1)


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

        plt.plot(iv_pair[0, :], iv_pair[1, :], '.-')
        plt.ylim([-500, 0])
        plt.xlim([-5, 6])

        plt.xlabel("voltage (V)")
        plt.ylabel("currnet (A/m^2)")
        plt.show()

    def test_mj_cell_iv(self):
        sq1_cell = SQCell(eg=1.87, cell_T=300, plug_in_term=rev_breakdown_diode)

        sq2_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=rev_breakdown_diode)

        sq3_cell = SQCell(eg=1.0, cell_T=300, plug_in_term=rev_breakdown_diode)

        tj_cell = MJCell([sq1_cell, sq2_cell, sq3_cell])
        tj_cell.set_input_spectrum(load_astm(("AM1.5d")))

        solved_mj_v, solved_mj_i = tj_cell.get_iv()

        # plot the I-V of sq1,sq2 and sq3
        volt = np.linspace(-3, 3, num=300)
        plt.plot(volt, sq1_cell.get_j_from_v(volt), label="top cell")
        plt.plot(volt, sq2_cell.get_j_from_v(volt), label="middle cell")
        plt.plot(volt, sq3_cell.get_j_from_v(volt), label="bottom cell")

        plt.plot(solved_mj_v, solved_mj_i, '.-', label="MJ cell")
        plt.ylim([-200, 0])
        plt.xlim([-5, 6])

        plt.xlabel("voltage (V)")
        plt.ylabel("currnet (A/m^2)")
        plt.legend()
        plt.show()

    def test_mj_j_from_v(self):
        sq1_cell = SQCell(eg=1.87, cell_T=300, plug_in_term=rev_diode)

        sq2_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=rev_diode)

        sq3_cell = SQCell(eg=1.0, cell_T=300, plug_in_term=rev_diode)

        tj_cell = MJCell([sq1_cell, sq2_cell, sq3_cell])
        tj_cell.set_input_spectrum(load_astm(("AM1.5d")))

        solved_mj_v, solved_mj_i = tj_cell.get_iv()

        volt = np.linspace(2.5, 5, num=300)
        solved_current = tj_cell.get_j_from_v(volt,max_iter=3)
        interped_i = np.interp(volt, solved_mj_v, solved_mj_i)

        print(solved_current - interped_i)

        plt.plot(volt, interped_i, '.-')
        plt.plot(solved_mj_v, solved_mj_i, '.-')
        plt.plot(volt, solved_current, '.-', label='get_j_from_v', alpha=0.3)

        plt.ylim([-200, 0])
        plt.legend()
        plt.show()


    def test_series_connected_mj_cells(self):
        sq1_cell = SQCell(eg=1.87, cell_T=300, plug_in_term=rev_breakdown_diode)

        sq2_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=rev_breakdown_diode)

        sq3_cell = SQCell(eg=1.0, cell_T=300, plug_in_term=rev_breakdown_diode)

        tj_cell = MJCell([sq1_cell, sq2_cell, sq3_cell])
        tj_cell.set_input_spectrum(load_astm(("AM1.5d")))

        connected_cells=[]
        for i in range(3):
            c_tj_cell=copy.deepcopy(tj_cell)
            multi=np.random.random()*0.1
            c_tj_cell.set_input_spectrum(load_astm("AM1.5d")*(1+multi))
            connected_cells.append(c_tj_cell)

        # Connect all the subcells in series
        iv_funcs=[]
        for mj_cells in connected_cells:
            for subcell in mj_cells.subcell:
                iv_funcs.append(subcell.get_j_from_v)

        from pypvcell.fom import isc
        for cell in connected_cells:
            v,i=cell.get_iv()
            print(isc(v,i))
            plt.plot(v,i)

        iv_pair = solve_series_connected_ivs(iv_funcs, vmin=-1, vmax=3.5, vnum=300)

        plt.plot(iv_pair[0], iv_pair[1],'.-')
        plt.ylim([-300,0])
        plt.show()


if __name__ == '__main__':
    unittest.main()
