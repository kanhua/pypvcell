import numpy as np
from scipy.interpolate import interp1d

def voc(voltage, current):
    """
    Calculate the open-circuit voltage of an I-V curve.

    :param voltage: the array of voltage
    :type voltage: np.ndarray
    :param current: the array of current
    :type current: np.ndarray
    :return: the value of Voc
    """

    assert isinstance(voltage, np.ndarray)
    assert isinstance(current, np.ndarray)

    interp = interp1d(x=current, y=voltage)

    voc_value = interp(0)

    if np.isnan(voc_value):
        raise ValueError("Voc not found")
    else:
        return voc_value


def isc(voltage, current):
    """
    Calculate the short-circuit current of an I-V curve

    :param voltage: voltage array
    :type voltage: np.ndarray
    :param current: current array
    :type current: np.ndarray
    :return: The value of isc
    """

    assert isinstance(voltage, np.ndarray)
    assert isinstance(current, np.ndarray)

    interp = interp1d(x=voltage, y=current)

    isc_value = interp(0)

    if np.isnan(isc_value):
        raise ValueError("Isc not found")
    else:
        return isc_value


def ff(voltage, current):
    """
    Calculate the fill factor from an I-V curve

    :param voltage: voltage array
    :type voltage: np.ndarray
    :param current: current array
    :type current: np.ndarray
    :return: fill factor (in numerics)
    """
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
    """
    Calculate the maximum power of an I-V curve.
    Note that this function uses the convention of negative Isc.

    :param voltage: voltage array
    :type voltage: np.ndarray
    :param current: current array
    :type current: np.ndarray
    :return: the value of maximum power
    """

    assert isinstance(voltage, np.ndarray)
    assert isinstance(current, np.ndarray)

    # Find maximum power
    power_val=voltage*(-current)
    max_power_val=np.max(power_val)

    return max_power_val