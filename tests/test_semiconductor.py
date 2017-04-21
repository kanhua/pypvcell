import unittest
import numpy as np
import scipy.constants as sc

from pypvcell.semiconductor import calc_jnp,n_intrinsic,builtin_volt
from pypvcell.spectrum import Spectrum


class SemiconductorTestCase(unittest.TestCase):
    def test_jnp(self):

        w_n=136.735*1e-9 #m
        w_p=10.939*1e-9     #m

        x_n=500*1e-9 #m
        x_p=100*1e-9 #m

        d_n=1.293e-3 # m^2/s
        d_p=8.79e-5 # m^2/s

        l_n=0.2e-6 #m
        l_p=0.1e-6 #m

        s_n=0
        s_p=0

        n_d=1e19*1e6 # /m^3
        n_a=1e17*1e6 # /m^3

        abs_file='./gaas_nkalpha.csv'
        abs_array=np.loadtxt(abs_file,delimiter=',',skiprows=1)

        abs = Spectrum(abs_array[:, 0], abs_array[:, 2]*100, x_unit='nm')

        abs_x,abs_y=abs.get_spectrum(to_x_unit='J')

        init_photon_flux=np.ones_like(abs_y)

        T=297
        Eg=1.42*sc.e

        m_e=0.061*sc.m_e
        m_h=0.341*sc.m_e

        ni=n_intrinsic(Eg,m_e,m_h,T)
        print(ni)

        vbi=builtin_volt(n_d,n_a,ni,T)
        print(vbi)

        Jgen, Jn, Jp, Jrec, bsInitial, energies, jgen, jn, jp=calc_jnp(V=0,Vbi=vbi,alphaBottom=abs_y,
                 alphaI=abs_y,alphaTop=abs_y,bsInitial=init_photon_flux,
                 bs_incident_on_top=init_photon_flux,d_bottom=d_n,d_top=d_p,energies=abs_x,
                 T=T,l_bottom=l_n,l_top=l_p,ni=ni,pn_or_np='pn',s_bottom=s_n,s_top=s_p,
                 w_bottom=w_n,w_top=w_p,x_bottom=x_n,x_top=x_p,xi=0)

        #print(Jgen/sc.e)
        #print(Jp/sc.e)
        print(jn[20:30]/sc.e)
        #print(energies/sc.e)

        import matplotlib.pyplot as plt

        plt.plot((jn+jp+jgen)/sc.e)
        plt.show()






if __name__ == '__main__':
    unittest.main()
