from ivsolver import gen_rec_iv_with_rs_by_newton
from fom import ff,voc
import numpy as np

import matplotlib.pyplot as plt

j01=5e-8
j02=0
n1=1
n2=2
temperature=300
rshunt=1e10

def get_rs(contact_rs):
    """

    :param contact_rs: in the unit of ohm.cm^2
    :return:
    """
    m_to_cm=100
    metal_coverage=0.34
    rs=contact_rs/m_to_cm**2/metal_coverage

    return rs


contact_res=[0.1,0.05,0.002]

#rs=1e-10
voltage=np.linspace(-0.8,1,num=50)

for r in contact_res:

    rs=get_rs(r)
    v,i=gen_rec_iv_with_rs_by_newton(j01, j02, n1, n2, temperature, rshunt=rshunt, rseries=rs,
                                     voltage=voltage, jsc=400,verbose=False)

    print("Rs:%s"%r)
    print("FF:%s"%ff(v,i))
    print("Voc:%s"%voc(v,i))

    plt.plot(v,-i,label="%s Ohm.cm^2"%r)

plt.legend()
plt.ylim([0,500])
plt.xlabel("voltage (V)")
plt.ylabel("current (A/m^2)")

print(rs)

plt.savefig("si_iv.png")
plt.show()
