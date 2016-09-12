from test_detailed_balance_MJ import DBMJTestCase
from test_spectrum import SpectrumTestCases
from test_fom import FoMTestCase
import unittest


if __name__=="__main__":

    suite=unittest.TestSuite()

    suite.addTest(FoMTestCase("test_voc"))
    suite.addTest(FoMTestCase("test_isc"))
    suite.addTest(FoMTestCase("test_ff"))
    suite.addTest(DBMJTestCase("test_rad_to_voc"))
    suite.addTest(SpectrumTestCases("test_1"))
    suite.addTest(SpectrumTestCases("test_2"))
    suite.addTest(SpectrumTestCases("test_7"))
    suite.addTest(SpectrumTestCases("test_8"))
    suite.addTest(SpectrumTestCases("test_9"))
    suite.addTest(SpectrumTestCases("test_mul_scalar"))
    suite.addTest(SpectrumTestCases("test_mul_spectrum"))

    unittest.TextTestRunner(verbosity=3).run(suite)

