__author__ = 'kanhua'

import numpy as np
import matplotlib.pyplot as plt

def band_gap(x):
    """
    Band gap of AlxGa(1-x)As
    source: http://www.ioffe.ru/SVA/NSM/Semicond/AlGaAs/bandstr.html
    :return: band gap x in eV
    """
    if x<=0.45:
        return 1.424+1.247*x
    else:
        return 1.9+0.125*x+0.143*np.power(x,2)




if __name__=="__main__":
    print(band_gap(0.22))

    x_grid=np.linspace(0,1,num=100)

    algaas_band_gap=np.array([band_gap(x) for x in x_grid])

    plt.plot(x_grid,algaas_band_gap)
    plt.show()