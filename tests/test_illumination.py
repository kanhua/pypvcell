import unittest
from illumination import Illumination


class MyTestCase(unittest.TestCase):
    def test_something(self):
        spec="AM1.5g"
        ill = Illumination(spectrum=spec)
        tp=ill.total_power()
        print("total power of {0}:{1}".format(spec,tp))


if __name__ == '__main__':
    unittest.main()
