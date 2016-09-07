import matplotlib.pyplot as plt

from ivsolver import calculate_bed
from spectrum_base_update import Spectrum
from photocurrent import gen_square_qe

import scipy.constants as sc
import numpy as np



sp=gen_square_qe(1,0.9)

x,y1=calculate_bed(sp,300)
x,y2=calculate_bed(gen_square_qe(1.9,0.9),300)

y1=y1*np.exp(sc.e*1.0/(sc.k*300))
y2=y2*np.exp(sc.e*1.0/(sc.k*300))

plt.semilogy(x,y1,hold=True)
plt.semilogy(x,y2)

plt.show()