import unittest
from test_detailed_balance_MJ import DBMJTestCase
from test_spectrum import SpectrumTestCases
from test_fom import FoMTestCase
from test_solarcell import SolarCellTestCase
from test_transfer_matrix import TMCase
from test_high_precision_cell import TestHighPrecisionSQCellCase

if __name__ == "__main__":

    test_cases = [DBMJTestCase(), SpectrumTestCases(),
                  FoMTestCase(), SolarCellTestCase(), TMCase(),
                  TestHighPrecisionSQCellCase()]

    for t in test_cases:
        suite = unittest.TestLoader().loadTestsFromModule(t)

        unittest.TextTestRunner().run(suite)
