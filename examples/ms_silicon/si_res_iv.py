from ivsolver import gen_rec_iv_with_rs_by_newton
from fom import ff, voc, max_power
import numpy as np

import matplotlib.pyplot as plt

from si_res import lump_r
from units_system import UnitsSystem

us = UnitsSystem()

j01 = 5e-9
j02 = 0
n1 = 1
n2 = 2
temperature = 300
rshunt = 1e10


def get_rs(contact_rs):
    """

    :param contact_rs: in the unit of ohm.cm^2
    :return:
    """
    m_to_cm = 100
    metal_coverage = 0.34
    rs = contact_rs / m_to_cm ** 2 / metal_coverage

    return rs


# contact_res=[0.1,0.05,0.002]

# contact_res=np.linspace(0.1,0.6,num=6)
contact_res = np.logspace(-3, 1, num=4)

# rs=1e-10
voltage = np.linspace(-0.8, 1, num=50)

for r in contact_res:
    # rs=get_rs(r)

    rs = us.siUnits(r, 'Ohm cm^2')

    lr, w = lump_r(us.siUnits(1, "cm"), us.siUnits(1, "cm"),
                   us.siUnits(25, 'um'), us.siUnits(100, 'nm'),
                   us.siUnits(2, "mOhm cm2"), us.siUnits(2.44e-8, "Ohm m"),
                   rho_sh=us.siUnits(0.0001, "Ohm"), N=7)

    # w=us.asUnit(w,'cm')


    v, i = gen_rec_iv_with_rs_by_newton(j01, j02, n1, n2, temperature, rshunt=rshunt, rseries=rs,
                                        voltage=voltage, jsc=100, verbose=False)

    mp = max_power(v, i)
    print("Rs:%s" % r)
    print("FF:%s" % ff(v, i))
    print("Voc:%s" % voc(v, i))

    print("max power %s" % mp)
    print("efficiency %s" % (mp / 1000))

    plt.plot(v, -i, label="%s Ohm.cm^2" % r)

plt.legend()
plt.ylim([0, 500])
plt.xlabel("voltage (V)")
plt.ylabel("current (A/m^2)")

print(rs)

plt.savefig("si_iv.png")
plt.show()
