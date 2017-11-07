============
Introduction
============

OffsetNFW is a package designed to quickly and accurately generate NFW lensing profiles, with easy
extensibility to add non-NFW profiles in the future.  It has direct computation of the basic
profiles through methods that end with ``_theory``, plus versions that rely on internal 
interpolation tables for fast computation, particularly of offset (miscentered) profiles, that is, 
the lensing profile when your observation has errors in the placement of the halo center.

Installation
============

OffsetNFW is a pure Python package, no compilation needed. You can install it with the command:

>>> python setup.py install

To run OffsetNFW, you must have:
 - astropy
 - numpy
 - scipy
OffsetNFW has only been tested on Python 2.7.  Support for Python 3 may be added in the future.

Support
=======

Any questions or bug reports should be reported via the GitHub issues page at 
`<https://github.com/msimet/OffsetNFW>`_.
