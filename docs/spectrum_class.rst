Spectrum
=========

Spectrum class is designed to simplify the unit conversions, interpolation and arithmetic operations of spectrum data in the form of ``y(x)`` or ``y(x)dx``.
When modeling solar cells, we often have to deal with spectrum with different units. For example, the ``x`` of your EQE is in photon energy (eV), but the ``x`` of the illumination spectrum is in wavelength (nm), and you want to multiply them to get Jsc. You then have to convert the ``x`` of EQE into wavelength, reverse the sequence of both ``x`` and ``y``, interpolate either the EQE or the illumination spectrum, convert ``y`` of the illumination spectrum into photon flux, multiply the data and finally run numerical integration to get Jsc. These processes can cause some headache and very error prone. ``Spectrum`` class aims to make these calculations robust and easy.

Spectrum class bundles the data of ``x``, ``y`` and their associated units together into an object. This design may look clumsy at first sight, but this helps reduce many potential errors that could happen when handling the unit conversions of the data. The reason is that the value of ``y`` is often coupled with the unit of ``x``. For example, converting the ``y`` data from energy flux into photon flux requires ``x``.




Spectrum class API
---------------

.. automodule:: pypvcell.spectrum
.. autoclass:: pypvcell.spectrum.Spectrum
:members:
