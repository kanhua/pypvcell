import numpy as np
import matplotlib.pyplot as plt

from pypvcell.solarcell import SQCell,MJCell,DBCell
from pypvcell.illumination import Illumination
from pypvcell.photocurrent import gen_step_qe

input_ill=Illumination("AM1.5g",concentration=1)
top_eg=np.linspace(1.6,2,num=100) # Set up range of top cell band gaps

eta=np.zeros(100) # Initialize an array for storing efficiencies
jsc_ratio=np.zeros_like(eta)
si_cell=SQCell(eg=1.12,cell_T=293,n_c=3.5,n_s=1)

#qe=gen_step_qe(teg,1)
#tc=DBCell(qe,rad_eta=1,T=293,n_c=3.5,n_s=1) # Set up top cell
tc=SQCell(1.70,cell_T=293)
mj=MJCell([tc, si_cell]) # Make multijunction cell by "streaming" the 1J cells
mj.set_input_spectrum(input_ill) # Set up the illumination
eta=mj.get_eta(verbose=1) # Store the calculated efficiency in an array
print(eta)