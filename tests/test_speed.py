import unittest
import numpy as np
from spectrum_base_update import Spectrum
from spectrum_base import spectrum_base
import scipy.constants as sc
from units_system import UnitsSystem
import time

us=UnitsSystem()



ts=time.time()
init_wl = np.linspace(300, 500, num=10)
init_spec = np.ones(init_wl.shape)

test_spec_base = Spectrum(init_wl, init_spec, 'nm', is_photon_flux=False)

for i in range(30000):
    wl, spec = test_spec_base.get_spectrum('nm', flux="photon", flux="photon")

    expect_spec = init_spec / (sc.h * sc.c / us.siUnits(init_wl, 'nm'))

    assert np.all(np.isclose(spec, expect_spec))

print("time spent:%s"%(time.time()-ts))



ts=time.time()

test_spec_base = spectrum_base()
init_wl = np.linspace(300, 500, num=10)
init_spec = np.ones(init_wl.shape)
test_spec_base.set_spectrum(init_wl, init_spec, 'nm', is_photon_flux=False)

for i in range(30000):
    spectrum = test_spec_base.get_spectrum('nm', flux="photon", flux="photon")

    expect_spec=init_spec/(sc.h*sc.c/us.siUnits(init_wl,'nm'))

    assert np.all(np.isclose(spectrum[:,1],expect_spec))

print("time spent:%s"%(time.time()-ts))