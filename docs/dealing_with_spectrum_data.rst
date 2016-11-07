
Dealing with spectrum data
==========================

This tutorial demonstrates how to use Spectrum class to do various
arithmetic operations of Spectrum. This demo uses the Jsc calculation as
an example, namely

.. raw:: latex

   \begin{equation}
   J_{sc}=\int \phi(E)QE(E) dE
   \end{equation}

where :math:`\phi` is the illumination spectrum in photon flux,
:math:`E` is the photon energy and :math:`QE` is the quantum efficiency.

.. code:: python

    %matplotlib inline
    import numpy as np
    import scipy.constants as sc
    import matplotlib.pyplot as plt
    from pypvcell.spectrum import Spectrum
    from pypvcell.illumination import Illumination
    from pypvcell.photocurrent import gen_step_qe_array

Quantum efficiency
------------------

We first use a function ``gen_step_qe_array`` to generate a quantum
efficiency spectrum. This spectrum is a step function with a cut-off at
the band gap of 1.42 eV.

.. code:: python

    qe=gen_step_qe_array(1.42,0.9)
    plt.plot(qe[:,0],qe[:,1])
    plt.xlabel('photon energy (eV)')
    plt.ylabel('QE')




.. parsed-literal::

    Text(0,0.5,'QE')




.. image:: dealing_with_spectrum_data_files/dealing_with_spectrum_data_5_1.png


``qe`` is a numpy array. The recommeneded way to handle it is converting
it to ``Spectrum`` class:

.. code:: python

    qe_sp=Spectrum(x_data=qe[:,0],y_data=qe[:,1],x_unit='eV')

Unit conversion
~~~~~~~~~~~~~~~

When we want to retrieve the value of ``qe_sp`` we have to specicify the
unit of the wavelength. For example, say, converting the wavelength to
nanometer:

.. code:: python

    qe=qe_sp.get_spectrum(to_x_unit='nm')
    plt.plot(qe[0,:],qe[1,:])
    plt.xlabel('wavelength (nm)')
    plt.ylabel('QE')
    plt.xlim([300,1100])




.. parsed-literal::

    (300, 1100)




.. image:: dealing_with_spectrum_data_files/dealing_with_spectrum_data_9_1.png


Arithmetic operation
~~~~~~~~~~~~~~~~~~~~

We can do arithmetic operation directly with Spectrum class such as

.. code:: python

    # Calulate the portion of "non-absorbed" photons, assuming QE is equivalent to absorptivity
    tr_sp=1-qe_sp

.. code:: python

    tr=tr_sp.get_spectrum(to_x_unit='nm')
    plt.plot(tr[0,:],tr[1,:])
    plt.xlabel('wavelength (nm)')
    plt.ylabel('QE')
    plt.xlim([300,1100])




.. parsed-literal::

    (300, 1100)




.. image:: dealing_with_spectrum_data_files/dealing_with_spectrum_data_12_1.png


Illumination spectrum
---------------------

pypvcell has a class Illumination that is inherited from ``Spectrum`` to
handle the illumination. It inherits all the capability of ``Spectrum``
but has several methods specifically for sun illumination.

Some default standard spectrum is embedded in the ``pypvcell``:

.. code:: python

    std_ill=Illumination("AM1.5g")

Show the values of the data

.. code:: python

    ill=std_ill.get_spectrum('nm')
    plt.plot(*ill)
    plt.xlabel("wavelength (nm)")
    plt.ylabel("intensity (W/m^2-nm)")




.. parsed-literal::

    Text(0,0.5,'intensity (W/m^2-nm)')




.. image:: dealing_with_spectrum_data_files/dealing_with_spectrum_data_17_1.png


Calcuate the total intensity in W/m^2

.. code:: python

    std_ill.total_power()




.. parsed-literal::

    1000.3706555734423



Unit conversion of illumination spectrum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It requires a bit of attention of converting spectrum that is in the
form of :math:`\phi(E)dE`, i.e., the value of integration is a
meaningful quantitfy such as total power. This has been also handled by
``Illumination`` class. In the following case, we convert the wavelength
to eV. Please note that the units of intensity is also changed to
W/m^2-eV.

.. code:: python

    ill=std_ill.get_spectrum('eV')
    plt.plot(*ill)
    plt.xlabel("wavelength (eV)")
    plt.ylabel("intensity (W/m^2-eV)")




.. parsed-literal::

    Text(0,0.5,'intensity (W/m^2-eV)')




.. image:: dealing_with_spectrum_data_files/dealing_with_spectrum_data_21_1.png


Spectrum multiplication
-----------------------

To calcualte the overall photocurrent, we have to calculate
:math:`\phi(E)QE(E) dE` first. This would involves some unit conversion
and interpolation between two spectrum. However, this is easily dealt by
``Spectrum`` class:

.. code:: python

    # calculate \phi(E)QE(E) dE. 
    # Spectrum class automatically convert the units and align the x-data by interpolating std_ill
    jsc_e=std_ill*qe_sp

Here's a more delicate point. We should convert the unit to photon flux
in order to calculate Jsc.

.. code:: python

    jsc_e_a=jsc_e.get_spectrum('nm',to_photon_flux=True)
    plt.plot(*jsc_e_a)
    plt.xlim([300,1100])




.. parsed-literal::

    (300, 1100)




.. image:: dealing_with_spectrum_data_files/dealing_with_spectrum_data_25_1.png


Integrate it yields the total photocurrent density in A/m^2

.. code:: python

    sc.e*np.trapz(y=jsc_e_a[1,:],x=jsc_e_a[0,:])




.. parsed-literal::

    289.05897743220532



In fact, ``pypvcell`` already provides a function ``calc_jsc()`` for
calculating Jsc from given spectrum and QE:

.. code:: python

    from pypvcell.photocurrent import calc_jsc
    calc_jsc(std_ill,qe_sp)




.. parsed-literal::

    289.05944839925456


