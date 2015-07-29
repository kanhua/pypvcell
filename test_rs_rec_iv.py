__author__ = 'kanhua'

import matplotlib.pyplot as plt
from solcore3.beta.ivsolver import gen_rec_iv, get_v_from_j, gen_rec_iv_with_rs_by_newton
from solcore3.beta.ivsolver import solve_mj_iv
import numpy as np
import solcore3.beta.fom
import scipy.constants as sc

I01_array = (1e-20, 1e-18, 1e-17)
I02_array = (1e-13, 1e-15, 1e-15)
jsc_array = (137,158,154)


def test_gen_rev_iv():
    testv = np.linspace(0, 1.5, num=100)
    _ , testj = gen_rec_iv(I01_array[0], I02_array[0], 1, 2, 300, 1e20, testv)

    plt.semilogy(testv, testj)
    plt.show()
    plt.close()

def test_gen_rec_iv_rs():
    testv = np.linspace(0.1, 1.1, num=100)
    testv_1 , testj_1 = gen_rec_iv_with_rs_by_reverse(I01_array[0], I02_array[0], 1, 2, 300, 1e20,150, testv) # 150 ohm/m^2

    _ , testj_no_rs = gen_rec_iv(I01_array[0], I02_array[0], 1, 2, 300, 1e20, testv)

    plt.semilogy(testv_1, testj_1,'o',testv,testj_no_rs,'o')
    plt.show()
    plt.close()


def test_gen_rec_iv_rs_2():
    testv = np.linspace(0, 1.6, num=50)
    testv_1 , testj_1 = gen_rec_iv_with_rs_by_newton(I01_array[0], I02_array[0], 1, 2, 300, 1e20,5e-6,testv,jsc=140000) # 150 ohm/m^2

    _ , testj_no_rs = gen_rec_iv(I01_array[0], I02_array[0], 1, 2, 300, 1e20, testv,jsc=140000)


    # get voc
    print(np.interp(0,testj_1,testv_1))
    print(np.interp(0,testj_no_rs,testv_1))


    #plt.plot(testv_1, testj_1,'o',testv,testj_no_rs,'o')
    #plt.ylim([-1000,0])
    #plt.xlim([0,10])
    plt.plot(testv_1,testj_1,'o-',testv,testj_no_rs,'o-')
    plt.ylim([-150000,100])
    plt.legend(["with resistance","no resistance"],loc='best')
    #print((testj_1-testj_no_rs)/testj_no_rs)
    #print(testj_1)
    #print(testj_no_rs)
    plt.show()
    plt.close()


def test_gen_rec_iv_rs_dark():
    testv = np.linspace(-1, 1.6, num=50)
    testv_1 , testj_1 = gen_rec_iv_with_rs_by_newton(I01_array[0], I02_array[0], 1, 2, 300, 1e20,0,testv,jsc=0) # 150 ohm/m^2

    _ , testj_no_rs = gen_rec_iv(I01_array[0], I02_array[0], 1, 2, 300, 1e20, testv,jsc=0)

    plt.semilogy(testv_1,testj_1,'o-',testv,testj_no_rs,'o-')
    plt.legend(["with resistance","no resistance"])
    plt.show()
    plt.close()
    print(testj_1)
    print(testj_no_rs)
    print(np.isclose(testj_1,testj_no_rs))
    assert np.allclose(testj_1,testj_no_rs,rtol=1e-3)



def test_trial_func():

    j01=I01_array[0]
    j02=I02_array[0]
    n1=1
    n2=2
    temperature=300
    rseries=0.1
    rshunt=1e20

    v=1.3

    jsc=0
    def f(current):
        result=current - (j01 * (np.exp(sc.e * (v-rseries*current) / (n1 * sc.k * temperature)) - 1)
                          + j02 * (np.exp(sc.e * (v-rseries*current) / (n2 * sc.k * temperature)) - 1) +
                          v / rshunt) + jsc
        return result

    def fp(current):
        result=1-(-rseries* j01 * (np.exp(sc.e * (v-rseries*current) / (n1 * sc.k * temperature)) - 1)
                  -rseries* j02 * (np.exp(sc.e * (v-rseries*current) / (n2 * sc.k * temperature)) - 1))
        return result

    test_current=np.linspace(-100,100,num=100)
    plt.plot(test_current,f(current=test_current))
    plt.ylim([-1,1])
    print(f(test_current))
    plt.show()


test_gen_rec_iv_rs_2()

#test_gen_rec_iv_rs_dark()

#test_trial_func()
