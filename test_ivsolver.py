import matplotlib.pyplot as plt
from ivsolver import gen_rec_iv, get_v_from_j
from ivsolver import solve_mj_iv
import numpy as np

I01_array = (1e-20, 1e-18, 1e-17)
I02_array = (1e-13, 1e-15, 1e-15)
jsc_array = (137,158,154)


def test_gen_rev_iv():
    testv = np.linspace(0, 1.5, num=100)
    _ , testj = gen_rec_iv(I01_array[0], I02_array[0], 1, 2, 300, 1e20, testv)

    plt.semilogy(testv, testj)
    plt.show()
    plt.close()


def test_get_v_from_j():
    testv = np.linspace(0, 1.5, num=100)
    _ , testj = gen_rec_iv(I01_array[0], I02_array[0], 1, 2, 300, 1e20, testv)

    print(get_v_from_j(testv, testj, 200))


def test_solve_mj_iv():
    voltage = np.linspace(-0.5, 1.4, num=200)
    mjlist = [gen_rec_iv(I01_array[i], I02_array[i], 1, 2, 300, 1e20, voltage) for i in range(3)]

    for mj in mjlist:
        plt.plot(mj[0],mj[1],hold=True)
    #plt.plot(mjlist[2][0],mjlist[2][1])
    #plt.show()

    v,i=solve_mj_iv(mjlist)

    plt.plot(v,i)
    plt.ylim((np.min(i)-abs(np.min(i))*0.1,np.max(i)*1.1))
    plt.show()

def test_solve_mj_iv_with_jsc():

    voltage = np.linspace(-0.5, 1.4, num=200)
    mjlist = []
    for i in range(3):
        mjlist.append(gen_rec_iv(I01_array[i], I02_array[i],
                                 1, 2, 300, 1e20, voltage, jsc=jsc_array[i]))

    for mj in mjlist:
        plt.plot(mj[0],mj[1],hold=True)
    #plt.plot(mjlist[2][0],mjlist[2][1])
    #plt.show()

    v,i=solve_mj_iv(mjlist)


    plt.plot(v,i)
    plt.ylim((-200,20))
    plt.show()

    print(fom.max_power(v,i)/900)



test_gen_rev_iv()
test_get_v_from_j()
test_solve_mj_iv()
test_solve_mj_iv_with_jsc()

