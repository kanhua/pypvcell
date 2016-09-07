from distutils.core import setup

setup(name='pypvcel',
      version='1.0',
      py_modules=['photocurrent','illumination','units_system'],
      data_files=[('./', ['astmg173.csv'])]
      )