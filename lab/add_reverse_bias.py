from pypvcell.solarcell import SQCell
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from pypvcell.illumination import load_astm


def rev_diode(voltage):
    rev_j01 = 4.46e-19
    rev_bd_v=20
    return -rev_j01*np.exp(sc.e*(-voltage-rev_bd_v)/(sc.k*300)-1)

sq1_cell=SQCell(eg=1.3,cell_T=300,plug_in_term=rev_diode)
sq1_cell.set_input_spectrum(load_astm("AM1.5d"))

test_v=np.linspace(-21,1.5,num=100)

test_j=sq1_cell.get_j_from_v(test_v)

print(test_j)

# model reverse breakdown

print(sq1_cell.j01)

plt.plot(test_v,test_j)

plt.show()
plt.close()

