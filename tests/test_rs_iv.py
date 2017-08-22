"""
This script test iv with series resistance
"""

__author__ = 'kanhua'

import unittest
import numpy as np
import matplotlib.pyplot as plt
from pypvcell.ivsolver import gen_rec_iv_with_rs_by_newton, gen_rec_iv


I01_array = (1e-20, 1e-18, 1e-17)
I02_array = (1e-13, 1e-15, 1e-15)
jsc_array = (137, 158, 154)


class MyTestCase(unittest.TestCase):
    def test_gen_rec_iv_rs_dark(self):
        """
        This test sets Rs=0 in the I-V solver and compares its results with gen_rec_iv, both with rs=0
        :return:
        """

        testv = np.linspace(-1, 1.6, num=50)
        testv, testj_rs = gen_rec_iv_with_rs_by_newton(I01_array[0], I02_array[0], 1, 2, 300, 1e20, 0, testv,
                                                       jsc=0)

        _, testj_no_rs = gen_rec_iv(I01_array[0], I02_array[0], 1, 2, 300, 1e20, testv, jsc=0)

        plt.semilogy(testv, testj_rs, 'o-', testv, testj_no_rs, 'o-')
        plt.legend(["with resistance", "no resistance"])
        #plt.show()
        plt.close()
        print(testj_rs)
        print(testj_no_rs)
        print(np.isclose(testj_rs, testj_no_rs))

        assert np.allclose(testj_rs, testj_no_rs, rtol=1e-3)


    def test_gen_iv_light(self):

        test_rs_list=[0,1.5e-6,5e-6] #At the moment, rs=1e-5 does not converge well

        for rs in test_rs_list:
            testv = np.linspace(0, 1.6, num=50)
            _, testj_1 = gen_rec_iv_with_rs_by_newton(I01_array[0], I02_array[0], 1, 2, 300, 1e20, rs, testv,
                                                      jsc=140000)

            _, testj_no_rs = gen_rec_iv(I01_array[0], I02_array[0], 1, 2, 300, 1e20, testv, jsc=140000)

            # get voc
            voc_rs = np.interp(0, testj_1, testv)
            voc2_nors = np.interp(0, testj_no_rs, testv)

            plt.plot(testv,testj_1,'o-',)

            assert np.isclose(voc_rs, voc2_nors, rtol=1e-2)

        plt.plot(testv, testj_1, 'o-', testv, testj_no_rs, 'o-')
        plt.ylim([-150000, 100])
        plt.legend(["with resistance", "no resistance"], loc='best')
        plt.close()

    @unittest.expectedFailure
    def test_aggressive_case(self):
        test_rs_list=[0,1e-7] #At the moment, rs=1e-5 does not converge well

        for rs in test_rs_list:
            testv = np.linspace(0, 1.6, num=50)
            _, testj_1 = gen_rec_iv_with_rs_by_newton(I01_array[0], I02_array[0], 1, 2, 300, 1e20, rs, testv,
                                                      jsc=140000)

            _, testj_no_rs = gen_rec_iv(I01_array[0], I02_array[0], 1, 2, 300, 1e20, testv, jsc=140000)

            # get voc
            voc_rs = np.interp(0, testj_1, testv)
            voc2_nors = np.interp(0, testj_no_rs, testv)

            plt.plot(testv,testj_1,'o-',)

            assert np.isclose(voc_rs, voc2_nors, rtol=1e-2)

        plt.plot(testv, testj_1, 'o-', testv, testj_no_rs, 'o-')
        plt.ylim([-150000, 100])
        plt.legend(["with resistance", "no resistance"], loc='best')
        plt.close()


if __name__ == '__main__':
    unittest.main()

