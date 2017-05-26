Pypvcell: A tool box for modeling solar cells
========================================================================

Pypvcell is collection of python functions for simulating the I-V characteristics of solar cell.
It is designed to be robust, extensible and easy-to-use.

Features
------------

* Written in pure python and its ecosystem.
* Innovative designs of APIs for rapid development and prototype your model.
* Extensively tested and validated.

Get the source
---------------

Source codes is hosted on `git <https://github.com/kanhua/pypvcell>`_ for `download <https://github.com/kanhua/pypvcell/archive/master.zip>`_.


Get Started
-----------------

.. toctree::
    :maxdepth: 1

    install
    tutorial


Overview of the model
-----------------------
.. image:: resources/pypvcell_architecture.png


Pypvcell APIs
-----------------

.. toctree::
    :maxdepth: 1

    spectrum_class
    photocurrent_m
    fom_m
    ivsolver
    transfer_matrix_optics


Highly experimental modules (not documented yet) in ``lab`` folder:

- `SMARTS interface <https://github.com/kanhua/pypvcell/tree/master/lab/SMARTS>`_
- `Analytical PIN solver <https://github.com/kanhua/pypvcell/blob/master/lab/analytical_pin.py>`_


Liscence:
--------------------------
`Apache 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_


Acknowledgement
--------------------------

The development of this software is partly supported by Japan New Energy and Industrial Technology Development Organization (NEDO).
Pypvcell is inspired by `Solcore3 <http://doi.org/10.1063/1.4822193>`_ by Imperial College London.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




