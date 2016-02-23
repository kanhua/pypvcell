"""This module calculates figures of merit of solar cell IVs"""
__author__ = 'kanhua'

import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt


def voc(voltage, current):
    assert isinstance(voltage, np.ndarray)
    assert isinstance(current, np.ndarray)

    #plt.plot(voltage,current)
    #plt.show()

    interp = interp1d(x=current, y=voltage)

    voc_value = interp(0)

    if np.isnan(voc_value):
        raise ValueError("Voc not found")
    else:
        return voc_value


def isc(voltage, current):
    assert isinstance(voltage, np.ndarray)
    assert isinstance(current, np.ndarray)

    interp = interp1d(x=voltage, y=current)

    isc_value = interp(0)

    if np.isnan(isc_value):
        raise ValueError("Isc not found")
    else:
        return isc_value


def ff(voltage, current):
    assert isinstance(voltage, np.ndarray)
    assert isinstance(current, np.ndarray)

    # get Isc and Voc
    isc_val=isc(voltage,current)
    voc_val=voc(voltage,current)

    # Find maximum power
    power_val=voltage*(-current)
    max_power_val=np.max(power_val)

    ff=max_power_val/(-isc_val*voc_val)

    return ff


def max_power(voltage,current):

    assert isinstance(voltage, np.ndarray)
    assert isinstance(current, np.ndarray)

    # Find maximum power
    power_val=voltage*(-current)
    max_power_val=np.max(power_val)

    return max_power_val