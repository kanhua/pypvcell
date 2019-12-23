import unittest
import numpy as np
import scipy.constants as sc
from scipy.optimize import bisect
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


def no_reverse_diode():
    sq1_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=None)
    sq1_cell.set_input_spectrum(load_astm("AM1.5d") * 0.5)

    sq2_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=None)
    sq2_cell.set_input_spectrum(load_astm("AM1.5d"))

    sq3_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=None)
    sq3_cell.set_input_spectrum(load_astm("AM1.5d") * 1.1)

    iv_funcs = [sq1_cell.get_j_from_v,
                sq2_cell.get_j_from_v,
                sq3_cell.get_j_from_v]

    volt = np.linspace(-1, 2, num=300)
    plt.plot(volt, sq1_cell.get_j_from_v(volt), '.-', label="cell 1")
    plt.plot(volt, sq2_cell.get_j_from_v(volt), '.-', label="cell 2")
    plt.plot(volt, sq3_cell.get_j_from_v(volt), '.-', label="cell 3")

    plt.ylim([-350, 0])
    plt.xlim([-1.5, 4])

    plt.xlabel("voltage (V)")
    plt.ylabel("current (A/m^2)")
    plt.legend()
    plt.savefig("./figs/no_rev_diode_iv.png", dpi=300)
    plt.savefig("./figs/no_rev_diode_iv.pdf")
    plt.show()

    plt.close()
    plt.figure()

    current_to_solve = sq1_cell.get_j_from_v(volt)
    current_to_solve = np.unique(current_to_solve)

    # plt.plot(volt, sq1_cell.get_j_from_v(volt),'.-',label="cell 1")
    sum_v = np.zeros_like(current_to_solve)
    for idx, iv_func in enumerate(iv_funcs[0:]):
        v, i = solve_v_from_j_by_bracket_root_finding(iv_func,
                                                      current_to_solve,
                                                      -3, 5, bisect)
        sum_v += v
        plt.plot(v, i, '.-', label="solved cell {}".format(idx + 1))

    # plot summed up the voltages
    plt.plot(sum_v, current_to_solve, '.-', label="connected cell")

    plt.ylim([-350, 0])
    plt.xlim([-1.5, 4])
    plt.xlabel("voltage (V)")
    plt.ylabel("current (A/m^2)")
    plt.legend()
    plt.savefig("./figs/no_rev_diode_iv_solved.png", dpi=300)
    plt.savefig("./figs/no_rev_diode_iv_solved.pdf")
    plt.show()


def with_reverse_diode(plot_arrow=False):
    sq1_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=rev_diode)
    sq1_cell.set_input_spectrum(load_astm("AM1.5d") * 0.5)

    sq2_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=rev_diode)
    sq2_cell.set_input_spectrum(load_astm("AM1.5d"))

    sq3_cell = SQCell(eg=1.42, cell_T=300, plug_in_term=rev_diode)
    sq3_cell.set_input_spectrum(load_astm("AM1.5d") * 1.1)

    iv_funcs = [sq1_cell.get_j_from_v,
                sq2_cell.get_j_from_v,
                sq3_cell.get_j_from_v]

    volt = np.linspace(-3, 2, num=300)
    arrow_dx = volt + 1.0
    arrow_dy = np.zeros_like(volt)
    plt.plot(volt, sq1_cell.get_j_from_v(volt), '.-', label="cell 1")
    prev_current = None

    if plot_arrow:
        for v in volt:
            current = sq1_cell.get_j_from_v(v)
            if (prev_current is None) or (np.isclose(current, prev_current, rtol=5e-2) == False):
                plt.arrow(v, sq1_cell.get_j_from_v(v), 2.0 - v, 0, color='C0', linestyle='--')
            prev_current = current

    plt.plot(volt, sq2_cell.get_j_from_v(volt), '.-', label="cell 2")

    prev_current = None
    if plot_arrow:
        for v in volt:
            current = sq2_cell.get_j_from_v(v)
            if (prev_current is None) or (np.isclose(current, prev_current, rtol=5e-2) == False):
                plt.arrow(v, sq2_cell.get_j_from_v(v), 2.0 - v, 0, color='C1', linestyle='--')
            prev_current = v

    plt.plot(volt, sq3_cell.get_j_from_v(volt), '.-', label="cell 3")

    prev_current = None
    if plot_arrow:
        for v in volt:
            current = sq3_cell.get_j_from_v(v)
            if (prev_current is None) or (np.isclose(current, prev_current, rtol=5e-2) == False):
                plt.arrow(v, sq3_cell.get_j_from_v(v), 2.0 - v, 0, color='C2', linestyle='--')
            prev_current = v

    plt.ylim([-350, 0])
    plt.xlim([-1.5, 4])

    plt.xlabel("voltage (V)")
    plt.ylabel("current (A/m^2)")
    plt.legend()
    plt.savefig("./figs/with_rev_diode_iv.png", dpi=600)
    plt.savefig("./figs/with_rev_diode_iv.pdf")
    plt.show()

    plt.close()
    plt.figure()

    solved_iv, subcell_ivs = solve_series_connected_ivs(iv_funcs, -2, 3, 100, return_subcell_iv=True, add_epsilon=0.0)

    # plot sub cell I-Vs
    for subcell_id in range(3):
        plt.plot(subcell_ivs[subcell_id, 0, :], subcell_ivs[subcell_id, 1, :], '.-',
                 label="cell {}".format(subcell_id + 1))

    # plot summed up the voltages
    plt.plot(solved_iv[0], solved_iv[1], '.-', label="connected cell")

    plt.ylim([-350, 0])
    plt.xlim([-3, 4])
    plt.xlabel("voltage (V)")
    plt.ylabel("current (A/m^2)")
    plt.legend()
    plt.savefig("./figs/with_rev_diode_iv_solved.png", dpi=600)
    plt.savefig("./figs/with_rev_diode_iv_solved.pdf")

    plt.show()

if __name__ == "__main__":
    no_reverse_diode()
    with_reverse_diode()
