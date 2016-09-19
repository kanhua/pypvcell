"""
Calculate the criteria of series resistance of silicon solar cell
"""

from units_system import UnitsSystem
import numpy as np

us = UnitsSystem()


def lump_r(sw, sl, fw, fh, rho_c, rho_f, rho_sh, N):
    w = sw / N

    pitch = w - fw  # pitch between fingers

    # print("pitch:%s"%pitch)

    Rc = rho_c / (fw * sl)

    Rf = rho_f * (sl / 2) / (fw * fh)

    Rsh = 2 * rho_sh * (sw / 2)

    R_lump = Rc + Rf + Rsh

    return R_lump, R_lump * w * sl


if __name__ == "__main__":
    lr, lr_per_area = lump_r(us.siUnits(1, "cm"), us.siUnits(1, "cm"),
                             us.siUnits(25, 'um'), us.siUnits(100, 'nm'),
                             us.siUnits(0.11, "Ohm cm2"), us.siUnits(2.44e-8, "Ohm m"),
                             rho_sh=us.siUnits(0.0001, "Ohm"), N=7)

    print("lumped resistance of a portion:%s" % lr)

    print(lr_per_area * 10000)

    lr, lr_per_area = lump_r(us.siUnits(15.6, "cm"), us.siUnits(15.6, "cm"),
                             us.siUnits(60, 'um'), us.siUnits(12, 'um'),
                             us.siUnits(2, "mOhm cm2"), us.siUnits(4.5, "uOhm cm"), rho_sh=us.siUnits(0.0, "Ohm"),
                             N=75)

    print(lr)

    print(lr_per_area * 10000)

    lr, lr_per_area = lump_r(us.siUnits(1, "cm"), us.siUnits(1, "cm"),
                             us.siUnits(25, 'um'), us.siUnits(100, 'nm'),
                             us.siUnits(0.11, "Ohm cm2"), us.siUnits(0, "Ohm m"),
                             rho_sh=us.siUnits(0.000, "Ohm"), N=7)

    print(lr, lr_per_area)

    print(lr_per_area * 10000)
