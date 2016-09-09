"""
Calculate the SQ-limit of 1-J cell
This calculation compares the file with the graph in etaOpt paper
"""


from detail_balanced_MJ import calc_1j_eta
import numpy as np
import matplotlib.pyplot as plt

etaopt_air_semi = np.loadtxt("../validation/eta_air_semi.txt", skiprows=1)

eg_array = np.linspace(0.5, 2.0)

eta = calc_1j_eta(eg=1.42, qe=1, r_eta=1, cell_temperature=300)

print(eta)

calc_method="eg"

#conc=1500*1000/918
conc=1500

eta_array_air_air = [calc_1j_eta(eg, qe=1,n_s=1,n_c=1 ,r_eta=1, concentration=conc,
                         cell_temperature=293, spectrum="AM1.5d",j01_method=calc_method) for eg in eg_array]

eta_array_air_semi= [calc_1j_eta(eg, qe=1, n_s=1,n_c=3.6,r_eta=1, concentration=conc,
                                cell_temperature=293, spectrum="AM1.5d",j01_method=calc_method) for eg in eg_array]

eta_array_semi_semi= [calc_1j_eta(eg, qe=1, n_s=3.6,n_c=3.6,r_eta=1, concentration=conc,
                                 cell_temperature=293, spectrum="AM1.5d",j01_method=calc_method) for eg in eg_array]


plt.plot(eg_array, eta_array_air_air,label="nc=1,ns=1",hold=True)
plt.plot(eg_array,eta_array_air_semi,label="nc=1,ns=3.6")
plt.plot(eg_array,eta_array_semi_semi,label="nc=3.6,ns=3.6")
plt.plot(etaopt_air_semi[:,0],etaopt_air_semi[:,1]/100,'o',label="etaopt_air_semi")
plt.xlabel("band gap (eV)")
plt.ylabel("efficiency")
plt.legend()
plt.grid()

plt.show()