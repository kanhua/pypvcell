from detail_balanced_MJ import calc_1j_eta
import numpy as np


etaopt_air_semi=np.loadtxt("./validation/eta_air_semi.txt",skiprows=1)

eg_array = np.linspace(0.5, 2.0)

eta = calc_1j_eta(eg=1.42, qe=1, r_eta=1, cell_temperature=300)

print(eta)




