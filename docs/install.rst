Installing Pypvcell
====================

Package requirements
--------------------------

- Pypvcell supports 3.x.
- Numpy, Scipy, pint, Matplotlib, jupyter


Setup your Python environment
------------------------------

1. Download `Anaconda <http://continuum.io/downloads>`_ or `Miniconda <https://conda.io/miniconda.html>`_. Read `this <https://conda.io/docs/download.html#should-i-download-anaconda-or-miniconda>`_ if you are not sure which one suit you best. Both Anaconda and Miniconda offer Python 2.7 and Python 3.x versions. To run pypvcell, you need **python 3.x**.

2. If you choose to install Miniconda, type the following script in Powershell or Terminal::

    conda install numpy scipy matplotlib jupyter pandas

3. You also have to install `pint <http://pint.readthedocs.io/en/0.8/>`_. This is a toolkit for doing unit conversion in python. You can install it by using conda forge::

    conda install -c conda-forge pint

or using pip::

    pip install pint

If you did not see error messages during the installations, you should be good to go.


Install Pypvcell via pip
-------------------------

We recommend a two step process:

First, download the repository with git clone::

    git clone https://github.com/kanhua/pypvcell.git

Switch to the downloaded Pypvcell directory, and install it via pip::

    pip install ./

If you would like to do some experimental changes of pypvcell, you can add parameter -e, i.e.,::

    pip install -e ./

In this way, instead of copying file to your python site-packages directory, pip creates a link to your downloaded pypvcell folder.
You can then directly play with the code in the pypvcell directory.

P.S.
We are planning to add this project to [the python package index](https://pypi.python.org/pypi) after a more steady version of pypvcell is released.
This would make the installation through PIP a little bit easier in the future.

Test the Installed Package
-----------------------------

In ``pypvcell/tests/`` folder, run::

    python3 all_utests.py


If all tests are passed, you should be able to run pypvcell properly on your machine.
If you see any issues, feel free to report it `here <https://github.com/kanhua/pypvcell/issues>`_.


What to Explore Next
-----------------------------

Run a jupyter notebook file in ``demos/`` folder. For instance::

    jupyter notebook ./demos/efficiency_vs_bandgap.ipynb


