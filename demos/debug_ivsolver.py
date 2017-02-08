import numpy as np
import matplotlib.pyplot as plt

from pypvcell.solarcell import SQCell,MJCell,DBCell
from pypvcell.illumination import Illumination
from pypvcell.photocurrent import gen_step_qe

ill=Illumination("AM1.5d",concentration=46200*1000/918)
mj=MJCell([SQCell(eg=1.84,cell_T=293,n_c=1,n_s=1),
           SQCell(eg=1.16,cell_T=293,n_c=1,n_s=1),
           SQCell(eg=0.69,cell_T=293,n_c=1,n_s=1)])
mj.set_input_spectrum(input_spectrum=ill)
print(mj.get_eta(verbose=1))