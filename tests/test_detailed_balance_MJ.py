import unittest
from photocurrent import gen_square_qe
from detail_balanced_MJ import rad_to_voc_fast,rad_to_voc
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_rad_to_voc(self):
        """
        Calculate the Voc from radiative efficiencies.
        The caluation is done by two different methods, and the results should be close

        :return:
        """

        test_qe = gen_square_qe(1.12, 0.8)

        r1 = rad_to_voc(0.001, test_qe)

        r2 = rad_to_voc_fast(0.001, test_qe)

        self.assertTrue(np.isclose(r1,r2,rtol=5e-3))


if __name__ == '__main__':
    unittest.main()
