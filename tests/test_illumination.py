import unittest
from pypvcell.illumination import load_astm


class MyTestCase(unittest.TestCase):
    def test_something(self):
        spec = "AM1.5g"
        ill = load_astm(spec)
        tp = ill.rsum()
        print("total power of {0}:{1}".format(spec, tp))
        self.assertAlmostEqual(tp,1000.370,places=2)

    def test_load_astm_fail(self):
        spec = "AM1.5j"
        with self.assertRaises(ValueError):
            ill = load_astm(spec)


if __name__ == '__main__':
    unittest.main()
