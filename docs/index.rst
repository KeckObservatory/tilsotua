.. tilsotua documentation master file, created by
   sphinx-quickstart on Thu Feb 20 00:31:49 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to tilsotua's documentation!
====================================

This code calculates the sky positions of slits within slitmasks designed for the Low-Resolution Imaging Spectrograph (LRIS) on the Keck I telescope.

This code uses the slit coordinates used with the UCO/Lick milling machine to produce the slit masks, reversing the steps taken by the `autoslit3 <https://www2.keck.hawaii.edu/inst/lris/autoslit_WMKO.html>`_ code employed by the original slitmask designers to calculate the WCS coordinates of the slits. These milling machine-specific slit vertices are held in FITS files created as an archive of slit designs by UCO/Lick or within the output files produced by autoslit (specifically the .file3 files). After reconstructing the slit WCS positions used by the original mask designers, the code also attempts a final astrometric correction (a simple shift) based on a comparison of the alignment star slits with the Gaia survey. (This correction can be turned off.)

These final WCS coordinates for the slits are recorded and output to a file, where they can then be used with the archive of LRIS observations.

If you use tilsotua in your work, please cite: `Sullivan et al. 2025 <https://ui.adsabs.harvard.edu/abs/2025PASP..137e4505S/abstract>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   xytowcs
   steps
   output
   tutorial
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
