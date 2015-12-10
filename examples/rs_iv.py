__author__ = 'kanhua'


from ivsolver import gen_rec_iv_with_rs_by_newton,calculate_j01
import matplotlib.pyplot as plt
import numpy as np

j01=calculate_j01(1.36,300,1)

v_range=np.linspace(-1,1.5,num=100)


v1,i1=gen_rec_iv_with_rs_by_newton(j01,0,1,2,300,1e10,1e-6,v_range)
v2,i2=gen_rec_iv_with_rs_by_newton(j01,0,1,2,300,1e10,1e-4,v_range)

plt.semilogy(v1,i1,hold=True)
plt.semilogy(v2,i2)
plt.xlabel("voltage")
plt.ylabel("current density (A/m^2)")
plt.legend(["0.01 Ohm-cm^2","1 Ohm-cm^2"],loc="best")
plt.grid()

plt.show()
