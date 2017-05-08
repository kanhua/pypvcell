import numpy
# from numpy import *
# from datetime import datetime
# from solcore3.units_system import UnitsSystem
# from numpy import flipud as reverse

import os, subprocess
import time
import pandas as pd

from SMARTS.smarts import get_clear_sky


times=pd.date_range(start='2016-12-20 04:00',end='2016-12-20 20:00',freq='H',tz='Japan')

df=get_clear_sky(times)

print(df.head())

d=pd.DatetimeIndex(times(0))
print(df.loc[d,:])
