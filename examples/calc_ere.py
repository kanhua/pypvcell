__author__ = 'kanhua'

import numpy as np

from photocurrent import calc_jsc
from illumination import illumination
from spectrum_base import spectrum_base
from detail_balanced_MJ import calc_ere
import matplotlib.pyplot as plt


def read_qe(input_file):

    eqe = np.loadtxt(input_file, delimiter="\t")

    test_qe = spectrum_base()

    test_qe.set_spectrum(eqe[:, 0], eqe[:, 1], "nm")

    return test_qe

def try_si_unsw():
    test_voc = 0.706
    eqe = np.loadtxt("SI UNSW EQE.txt", delimiter="\t")

    test_qe = spectrum_base()

    test_qe.set_spectrum(eqe[:, 0], eqe[:, 1], "nm")

    jsc = calc_jsc(illumination("AM1.5g", y_data=1, x_unit=null), test_qe)

    print("jsc %s:" % jsc)

    ere = calc_ere(test_qe, test_voc)

    return ere


def try_yamaguchi_algaas():
    test_voc = 1.28
    eqe = np.loadtxt("AlGaAs750deg.txt", delimiter="\t")

    test_qe = spectrum_base()

    test_qe.set_spectrum(eqe[:, 0], eqe[:, 1], "nm")

    ere = calc_ere(test_qe, test_voc)

    return ere


def guess_2J_rad():

    top_eqe=read_qe("topcell.txt")
    bot_eqe=read_qe("bottomcell.txt")

    total_voc=2.385

    top_voc_range=np.linspace(total_voc/2,1.6, num=100)

    top_ere_range=np.array([calc_ere(top_eqe,t) for t in top_voc_range])

    bot_ere_range=np.array([calc_ere(bot_eqe,t) for t in total_voc-top_voc_range])

    plt.semilogy(top_voc_range,top_ere_range,hold=True,label="top")
    plt.semilogy(top_voc_range,bot_ere_range,label="bottom")
    plt.legend(loc="best")
    plt.show()

    return




if __name__ == "__main__":
    print(try_si_unsw())

    print(try_yamaguchi_algaas())

    guess_2J_rad()



