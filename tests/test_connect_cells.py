"""
This unit test tests the cells connected in series or parallel


"""
import unittest
import numpy as np
from pypvcell.solarcell import ResistorCell, SeriesConnect, ParallelConnect


class SPTestCase(unittest.TestCase):

    def setUp(self):
        self.r1 = 1.0
        self.r2 = 2.0
        self.r1cell = ResistorCell(self.r1)
        self.r2cell = ResistorCell(self.r2)

    def test_parallel(self):
        pc = ParallelConnect([self.r1cell, self.r2cell])

        test_c = np.linspace(0.1, 2, 10)
        test_v = pc.get_v_from_j(test_c)

        expected_r = 1 / (1 / self.r1 + 1 / self.r2)

        calc_r = np.mean(test_v / test_c)

        self.assertTrue(np.isclose(expected_r, calc_r))

    def test_series(self):
        sc = SeriesConnect([self.r1cell, self.r2cell])

        test_c = np.linspace(0.1, 2, 10)
        test_v = sc.get_v_from_j(test_c)

        expected_r = self.r1 + self.r2

        calc_r = np.mean(test_v / test_c)

        self.assertTrue(np.isclose(expected_r, calc_r))


if __name__ == '__main__':
    unittest.main()
