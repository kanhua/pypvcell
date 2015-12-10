"""
detalied balance calculation of III-V on Si

"""

import matplotlib.pyplot as plt
import numpy as np
from solcore3.beta.detail_balanced_MJ import calc_mj_eta
import yaml

def calc_4j_eta(top_band_gap, top_band_eqe, concentration):
    cell_temperature = 300
    subcell_eg = np.array([top_band_gap, 1.12])
    subcell_qe = np.array([top_band_eqe, 1])
    subcell_rad_eff = np.array([1, 1])

    return calc_mj_eta(subcell_eg,subcell_qe,subcell_rad_eff,cell_temperature,concentration=concentration)

test_band_gap=np.linspace(1.2,2.1,num=100,endpoint=True)
test_qe=np.linspace(0.5,1,num=100,endpoint=True)

eta_array_1eV_1sun = [calc_4j_eta(eg, eqe, concentration=1) for eg in test_band_gap for eqe in test_qe]

eta_mtx=np.array(eta_array_1eV_1sun).reshape((len(test_band_gap),len(test_qe)))

yaml.dump([test_band_gap,test_qe,eta_mtx],open("III-V_si_2J.yaml",'w'))

plt.pcolormesh(test_qe,test_band_gap,eta_mtx)

plt.show()