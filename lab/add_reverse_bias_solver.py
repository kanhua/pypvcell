from pypvcell.solarcell import SQCell
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from pypvcell.illumination import load_astm


def find_voltage(v,i):
    pass


def rev_diode(voltage):
    rev_j01 = 4.46e-19
    rev_bd_v=20
    return -rev_j01*np.exp(sc.e*(-voltage-rev_bd_v)/(sc.k*300)-1)

sq1_cell=SQCell(eg=1.3,cell_T=300,plug_in_term=rev_diode)
sq1_cell.set_input_spectrum(load_astm("AM1.5d"))

sq2_cell=SQCell(eg=2.0,cell_T=300,plug_in_term=rev_diode)
sq2_cell.set_input_spectrum(load_astm("AM1.5d"))

test_v=np.linspace(-22,1.5,num=50)

print(sq1_cell.jsc)
print(sq2_cell.jsc)

test_j1=sq1_cell.get_j_from_v(test_v)
test_j2=sq2_cell.get_j_from_v(test_v)

def f1(x):
    return sq2_cell.get_j_from_v(x)+121

#current-limiting cell sq2_cell
from scipy.optimize import bisect
#zero_result=bisect(f1,0,1.5)
#print(zero_result)

#print(sq1_cell.get_j_from_v(zero_result))
print(f1(0))

zero_result=bisect(f1,0,-30)
print(zero_result)


#print(test_j1)
#print(test_j2)

# model reverse breakdown

#print(sq1_cell.j01)

#plt.plot(test_v,test_j1)
plt.plot(test_v,test_j2)

plt.show()
plt.close()

