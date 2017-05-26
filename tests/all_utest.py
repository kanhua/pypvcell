import unittest
from test_detailed_balance_MJ import DBMJTestCase
from test_spectrum import SpectrumTestCases
from test_fom import FoMTestCase
from test_solarcell import SolarCellTestCase


if __name__=="__main__":

    test_cases=[DBMJTestCase(),SpectrumTestCases(),FoMTestCase(),SolarCellTestCase()]

    for t in test_cases:
        suite = unittest.TestLoader().loadTestsFromModule(t)

        unittest.TextTestRunner().run(suite)

