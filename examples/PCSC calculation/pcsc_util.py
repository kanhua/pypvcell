import numpy as np
import matplotlib.pyplot as plt
from pypvcell.solarcell import SQCell
from pypvcell.illumination import Illumination


def calc_tp(input_ill: Illumination,
            total_area,
            a_3, a_c, f_c,
            eta_s, eta_3):
    """
    Calculate the total output power of a III-V-on-silicon mechanical stack cell

    :param input_ill: Input illumination spectrum
    :param total_area:
    :param a_3: the area of III-V cell
    :param a_c: The concetrated area
    :param f_c: The fraction of intensity that hits the concentrated area
    :param eta_s: function that calculates the efficiency of silicon cell
    :param eta_3: function that calculates the efficiency of III-V cell
    :return: total power
    """

    band_gap = 1.7
    sqcell = SQCell(band_gap, 300)
    sqcell.set_input_spectrum(input_ill)

    filtered_ill = sqcell.get_transmit_spectrum()

    if a_c >= a_3:
        # power of silicon underneath III-V
        p_3s = filtered_ill.total_power() * total_area * a_3 * eta_s(filtered_ill * float(f_c / a_c))

        # power of silicon in the concentrated area
        p_s_1 = input_ill.total_power() * total_area * (a_c - a_3) * eta_s(input_ill * float(f_c / a_c))

        # power of silicon in the diluted area
        p_s_2 = input_ill.total_power() * total_area * (1 - a_c) * eta_s(input_ill * float((1 - f_c) / (1 - a_c)))

        # power of III-V
        p_3 = input_ill.total_power() * total_area * a_3 * eta_3(input_ill * float(f_c / a_c))
        tp = p_3s + p_s_1 + p_s_2 + p_3

    else:

        # power of III-V in diluted area
        p_3_1 = input_ill.total_power() * total_area * (a_3 - a_c) * eta_3(input_ill * float((1 - f_c) / (1 - a_c)))

        # power of III-V in concentrated area
        p_3_2 = input_ill.total_power() * total_area * a_3 * eta_3(input_ill * float(f_c / a_c))

        # power of silicon underneath III-V
        p_3s_1 = filtered_ill.total_power() * total_area * a_3 * eta_s(filtered_ill * float(f_c / a_c))
        p_3s_2 = filtered_ill.total_power() * total_area * (a_3 - a_c) * eta_s(
            filtered_ill * float((1 - f_c) / (1 - a_c)))

        # power of silicon in diluted area
        p_s_1 = input_ill.total_power() * total_area * (1 - a_3) * eta_s(input_ill * float((1 - f_c) / (1 - a_c)))

        tp = p_3_1 + p_3_2 + p_3s_1 + p_3s_2 + p_s_1

    return tp


def eta_s(ill):
    """
    A function that calculates the efficiency of the silicon cell
    """
    band_gap = 1.1
    sqcell = SQCell(band_gap, 300)
    sqcell.set_input_spectrum(ill)

    return sqcell.get_eta()


def eta_3(ill):
    """
    A function that calculates the efficiency of the III-V solar cell from given spectrum
    """

    band_gap = 1.7
    sqcell = SQCell(band_gap, 300)
    sqcell.set_input_spectrum(ill)

    return sqcell.get_eta()


std_ill = Illumination("AM1.5g", concentration=1)

a_3 = 0.1
a_c = 0.2
f_c_arr = np.linspace(0.1, 0.9, num=20)

tp_arr = []
for f in f_c_arr:
    tp = calc_tp(std_ill, 1, a_3=a_3, a_c=a_c, f_c=f, eta_s=eta_s, eta_3=eta_3)
    tp_arr.append(tp)

tp_arr = np.array(tp_arr)

plt.plot(f_c_arr, tp_arr)
plt.show()
