import unittest
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import copy

from pypvcell.ivsolver import solve_series_connected_ivs, \
    solve_v_from_j_by_bracket_root_finding, solve_parallel_connected_ivs
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
        """
        Test five series-connected, non-uniform illuminated, Eg=1.42 solar cells

        :return:
        """

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
        """
        Test solving multi-junction cells, by breaking it down into series-connected subcells.

        :return:
        """
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
        """
        Test MJCell.get_j_from_v()

        :return:
        """
        sq1_cell = SQCell(eg=1.87, cell_T=300, plug_in_term=rev_diode)

        sq2_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=rev_diode)

        sq3_cell = SQCell(eg=1.0, cell_T=300, plug_in_term=rev_diode)

        tj_cell = MJCell([sq1_cell, sq2_cell, sq3_cell])
        tj_cell.set_input_spectrum(load_astm(("AM1.5d")))

        solved_mj_v, solved_mj_i = tj_cell.get_iv()

        volt = np.linspace(2.5, 5, num=300)
        solved_current = tj_cell.get_j_from_v(volt, max_iter=3)
        interped_i = np.interp(volt, solved_mj_v, solved_mj_i)

        print(solved_current - interped_i)

        plt.plot(volt, interped_i, '.-')
        plt.plot(solved_mj_v, solved_mj_i, '.-')
        plt.plot(volt, solved_current, '.-', label='get_j_from_v', alpha=0.3)

        plt.ylim([-200, 0])
        plt.legend()
        plt.show()

    def test_two_connected_mj_cells(self):
        """
        Test three series-connected triple-junction cells.
        We use MJCells.get_j_from_v() as the iv_func of the bracket-based root-finding solver.
        Warning: this is slow.

        :return:
        """

        sq1_cell = SQCell(eg=1.87, cell_T=300, plug_in_term=rev_diode)

        sq2_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=rev_diode)

        sq3_cell = SQCell(eg=1.0, cell_T=300, plug_in_term=rev_diode)

        tj_cell = MJCell([sq1_cell, sq2_cell, sq3_cell])
        tj_cell.set_input_spectrum(load_astm(("AM1.5d")))

        tj_cell_1 = copy.deepcopy(tj_cell)
        tj_cell_2 = copy.deepcopy(tj_cell)

        iv_funcs = [tj_cell_1.get_j_from_v, tj_cell_2.get_j_from_v]
        # solved_v,solved_i=solve_series_connected_ivs(iv_funcs, vmin=-2, vmax=3, vnum=10)
        volt_range = np.linspace(-1, 4, num=25)
        curr_range = tj_cell_1.get_j_from_v(volt_range)

        from scipy.optimize import brentq

        solved_v, solved_i = solve_v_from_j_by_bracket_root_finding(iv_funcs[0], curr_range, -5, 5, brentq)

        plt.plot(solved_v, solved_i, '.-')
        plt.ylim([-300, 0])
        plt.show()

    def test_series_connected_mj_cells(self):
        """
        Test three series-connected triple-junction cells.
        We break them down into nine series-connected subcells and perform ```solve_series_connected_ivs()```

        :return:
        """
        sq1_cell = SQCell(eg=1.87, cell_T=300, plug_in_term=rev_breakdown_diode)

        sq2_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=rev_breakdown_diode)

        sq3_cell = SQCell(eg=1.0, cell_T=300, plug_in_term=rev_breakdown_diode)

        tj_cell = MJCell([sq1_cell, sq2_cell, sq3_cell])
        tj_cell.set_input_spectrum(load_astm(("AM1.5d")))

        # Set up the MJCell() first. The purpose is only for calculating the transmitted spectrum
        # and the photocurrent of each subcell. We don't use the MJCell.get_j_from_v() to solve the I-V
        connected_cells = []
        number_of_mjcell = 3
        for i in range(number_of_mjcell):
            # Need copy.deepcopy to also copy the subcells
            c_tj_cell = copy.deepcopy(tj_cell)
            multi = np.random.random() * 0.1
            c_tj_cell.set_input_spectrum(load_astm("AM1.5d") * (1 + multi))
            connected_cells.append(c_tj_cell)

        # Extract each subcell, make them into series-connected cells
        # Connect all the subcells in series
        iv_funcs = []
        for mj_cells in connected_cells:
            for subcell in mj_cells.subcell:
                iv_funcs.append(subcell.get_j_from_v)

        from pypvcell.fom import isc
        for cell in connected_cells:
            v, i = cell.get_iv()
            print(isc(v, i))
            plt.plot(v, i)

        iv_pair = solve_series_connected_ivs(iv_funcs, vmin=-1, vmax=3.5, vnum=300)

        plt.plot(iv_pair[0], iv_pair[1], '.-')
        plt.ylim([-200, 0])
        plt.show()

    def test_parallel_connected_mj_cells(self):
        """
        Connected two triple-junction cells in parallel

        :return:
        """

        sq1_cell = SQCell(eg=1.87, cell_T=300, plug_in_term=rev_breakdown_diode)

        sq2_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=rev_breakdown_diode)

        sq3_cell = SQCell(eg=1.0, cell_T=300, plug_in_term=rev_breakdown_diode)

        tj_cell = MJCell([sq1_cell, sq2_cell, sq3_cell])
        tj_cell.set_input_spectrum(load_astm(("AM1.5d")))

        tj_cell_1 = copy.deepcopy(tj_cell)
        tj_cell_2 = copy.deepcopy(tj_cell)

        iv_funcs = [tj_cell_1.get_j_from_v, tj_cell_2.get_j_from_v]

        parallel_v, parallel_i = solve_parallel_connected_ivs(iv_funcs, vmin=-2, vmax=3.5, vnum=30)
        volt = np.linspace(-2, 3.5, num=30)
        curr = tj_cell_1.get_j_from_v(volt)

        plt.plot(volt, curr, '.-', label="single string")
        plt.plot(parallel_v, parallel_i, '.-', label="strings connected in parallel")
        plt.ylim([-600, 0])
        plt.show()

    def test_parallel_connected_mj_cells_2(self):
        """
        Try connecting two strings of cells. All are Eg=1.42 SQCell.
        Each string has five cells. One string has a cell has only 0.5 sun.

        :return:
        """

        normal_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=rev_diode)
        normal_cell.set_input_spectrum(load_astm("AM1.5d"))

        low_current_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=rev_diode)
        low_current_cell.set_input_spectrum(load_astm("AM1.5d") * 0.5)

        # A trick is used here to configure MJCell. The MJCell is not illuminated.
        # But each of the subcell is illuminated individually.
        tj_cell_1 = MJCell([copy.deepcopy(normal_cell) for i in range(4)] + [copy.deepcopy(low_current_cell)])

        tj_cell_2 = MJCell([copy.deepcopy(normal_cell) for i in range(5)])

        iv_funcs = [tj_cell_1.get_j_from_v, tj_cell_2.get_j_from_v]

        parallel_v, parallel_i = solve_parallel_connected_ivs(iv_funcs, vmin=-2, vmax=6, vnum=30)
        volt = np.linspace(-2, 5, num=30)
        curr_1 = tj_cell_1.get_j_from_v(volt)
        curr_2 = tj_cell_2.get_j_from_v(volt)

        plt.plot(volt, curr_1, '.-', label="string 1")
        plt.plot(volt, curr_2, '.-', label="string 2")
        plt.plot(parallel_v, parallel_i, '.-', label="parallel-connected string")
        plt.ylim([-600, 0])
        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
