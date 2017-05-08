import unittest
from SMARTS.smarts import parse_smarts_out


class MyTestCase(unittest.TestCase):
    def test_parse_smarts_out(self):
        sm_file='smarts295_dummy.out.txt'
        result=parse_smarts_out(file=sm_file,to_df=False)
        print(result)
        self.assertEqual(result['direct_tilt'],844.45)
        self.assertEqual(result['zenith'],48.236)
        self.assertEqual(result['azimuth'],180.0)
        self.assertEqual(result['direct_normal'],860.95)


if __name__ == '__main__':
    unittest.main()
