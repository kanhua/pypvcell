Pypvcell: An extensible programming framework for modeling solar cell
========================================================================

Pypvcell is collection of python functions for simulating the I-V characteristics of solar cell.
It is designed to be robust, extensible, easy-to-use framework for modeling the I-V characteristics of solar cells.

Features
------------


* PureWritten in pure python and its ecosystem
* Adopt innovative designs to simplify the codes
* Documented and tested
*


Overview of the model
-----------------------
.. image:: ./resources/pypvcell_architecture.png



Installation and Download
--------------------------
See https://github.com/kanhua/pypvcell


Examples
-----------

In Pypvcell, a type of solar cell is defined as class. For example, here's a way to define a solar cell class.
The following code demonstrates how to set up a solar cell at Shockley-Queisser limit:

.. code-block:: python

    from pypvcell.illumination import Illumination
    from pypvcell.solarcell import SQCell

    #Setup the illumination
    input_ill = Illumination("AM1.5g", concentration=1)

    # Setup the solar cell at a band gap of 1.42 eV
    sq_cell = SQCell(eg=1.42, cell_T=293, n_c=1, n_s=1)
    sq_cell.set_input_spectrum(input_spectrum=input_ill)

    # Print out the efficiency
    print(sq_cell.get_eta())



Prerequisites
--------------------------

- Pypvcell support both Python 2.x and 3. However, we recommend using Python 3.
- Numpy, Scipy, Matplotlib



Liscence:
--------------------------
MIT


Acknowledgement
--------------------------

The development of this software is partly suppored by Japan New Energy and Industrial Technology Development Organization (NEDO).
Pypvcell is inspired by Solcore by Markus Furher et al.


Modules
-----------------

.. toctree::
    :maxdepth: 2

    spectrum_class
    photocurrent_m
    fom_m




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




