__author__ = 'kanhua'

import numpy as np

from photocurrent import calc_jsc
from illumination import illumination
from spectrum_base import spectrum_base
from detail_balanced_MJ import calc_ere


def try_si_unsw():
    test_voc = 0.706
    eqe = np.loadtxt("SI UNSW EQE.txt", delimiter="\t")

    test_qe = spectrum_base()

    test_qe.set_spectrum(eqe[:, 0], eqe[:, 1], "nm")

    jsc = calc_jsc(illumination("AM1.5g", concentration=1), test_qe)

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


if __name__ == "__main__":
    print(try_si_unsw())

    print(try_yamaguchi_algaas())



