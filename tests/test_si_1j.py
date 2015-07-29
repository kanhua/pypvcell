"""
This scripts compares the efficiencies of Si 1J cell using J01 calculated by different methods
"""

__author__ = 'kanhua'

import unittest
from spectrum_base import spectrum_base
import numpy as np
from photocurrent import gen_qe_from_abs,gen_square_qe
from ivsolver import calculate_j01,calculate_j01_from_qe,gen_rec_iv
from detail_balanced_MJ import calc_mj_eta
from photocurrent import calc_jsc
from illumination import illumination
import fom
import matplotlib.pyplot as plt

class MyTestCase(unittest.TestCase):

    def setUp(self):

        abs_file="si_alpha.csv"
        self.si_alpha=np.loadtxt(abs_file,delimiter=',')
        self.si_alpha_sp=spectrum_base()
        self.si_alpha_sp.set_spectrum(self.si_alpha[:,0],self.si_alpha[:,1],wavelength_unit='m')

        self.direct_bg_qe=gen_square_qe(1.1,1)
        self.direct_bg_jsc=calc_jsc(illumination(),self.direct_bg_qe)

        self.layer_set=np.logspace(-6,-3,num=20)

        self.jsc_arr=[]
        for i in self.layer_set:
            qe=gen_qe_from_abs(self.si_alpha_sp,i)
            jsc=calc_jsc(illumination(),qe)
            self.jsc_arr.append(jsc)

        self.jsc_arr=np.array(self.jsc_arr)


    def test_calc_jsc(self):


        plt.semilogx(self.layer_set*1e6,self.jsc_arr,hold=True,label="from absorption coefficient")
        plt.semilogx([min(self.layer_set)*1e6,max(self.layer_set*1e6)],[self.direct_bg_jsc,self.direct_bg_jsc],label="assuming direct band gap")
        plt.legend(loc='best')
        plt.xlabel("layer thickness (um)")
        plt.ylabel("jsc (A/m^2)")
        plt.grid()
        plt.savefig("Si_layer_dep_jsc.pdf")
        plt.close()

        #print(jsc)
        #print(illumination().total_power())

    def test_calc_voc(self):

        self.voc_arr=[]
        self.eta_arr=[]
        test_v=np.linspace(-0.5,2,num=200)
        for i,t in enumerate(self.layer_set):

            qe=gen_qe_from_abs(self.si_alpha_sp,t)
            j01=calculate_j01_from_qe(qe)
            v,i=gen_rec_iv(j01,0,1,2,300,1e10,test_v,jsc=self.jsc_arr[i])
            self.eta_arr.append(fom.max_power(v,i)/illumination().total_power())
            self.voc_arr.append(fom.voc(v,i))

        plt.semilogx(self.layer_set*1e6,self.voc_arr)
        plt.xlabel("layer thickness (um)")
        plt.ylabel("Voc (V)")
        plt.grid()
        plt.savefig("Si_layer_dep_Voc.pdf")
        plt.close()

    def test_calc_eta(self):


        self.eta_arr=[]
        test_v=np.linspace(-0.5,2,num=200)
        for i,t in enumerate(self.layer_set):
            qe=gen_qe_from_abs(self.si_alpha_sp,t)
            j01=calculate_j01_from_qe(qe)
            v,i=gen_rec_iv(j01,0,1,2,300,1e10,test_v,jsc=self.jsc_arr[i])
            self.eta_arr.append(fom.max_power(v,i)/illumination().total_power())

        plt.semilogx(self.layer_set*1e6,self.eta_arr)
        plt.xlabel("layer thickness (um)")
        plt.ylabel("efficiency")
        plt.grid()
        plt.savefig("Si_layer_dep_eta.pdf")
        plt.close()


    def test_eta(self):

        si_layer=200e-6
        abs_file="si_alpha.csv"


        qe=gen_qe_from_abs(self.si_alpha_sp,si_layer)

        qe2=gen_square_qe(1.1,1,qe_below_edge=0)

        j01=calculate_j01_from_qe(qe2)

        j01_2=calculate_j01(1.1,300,1)


        test_voltage = np.linspace(-0.5, 1.5, num=300)

        v,i=gen_rec_iv(j01,0,1,2,300,1e10,test_voltage)


        eta=calc_mj_eta([1.42,1.1],[1,1],[1,1],300,replace_iv=(1,(v,i)))


        print(eta)

if __name__ == '__main__':
    unittest.main()
