===========
Basic Usage
===========

.. toctree::
   :maxdepth: 2

Setup
=====

All usage of OffsetNFW starts by making an instance of the NFWModel class.  This requires an 
instance of an ``astropy.cosmology`` object in order to perform computations such as the critical
surface mass density `\Sigma_{cr}`. So any script will likely begin:

>>> import offset_nfw
>>> import astropy.cosmology
>>> cosmology = astropy.cosmology.Planck15
>>> model = offset_nfw.NFWModel(cosmology)

There are several keyword arguments that can control the behavior of the ``NFWModel`` instance; 
these can be found in the :class:`offset_nfw.NFWModel` documentation.  The most common one you're
likely to use is the `dir` argument, which tells the object where the interpolation tables are
stored:

>>> model = offset_nfw.NFWModel(cosmology, '~/nfw_tables')

By default, the object will look in the current directory if tables are needed.

Available lensing quantities
============================

Once it has been created, the `NFWModel` offers several lensing quantities.
- :math:`\Sigma`, the surface mass density. Methods begin with `sigma`.
- :math:`\Delta\Sigma`, the differential surface mass density. Methods begin with `deltasigma`.
- :math:`\kappa`, the surface mass density as a fraction of `\Sigma_{cr}`. Methods begin with 
  `kappa`.
- :math:`\gamma_t`, the tangential shear. Methods begin with `gamma`.
- :math:`g`, the reduced shear (observed shear). Methods begin with `g`.
- :math:`\Upsilon`, the annular differential surface density (ADSD) statistics of Baldauf et al
  2010 and Mandelbaum et al 2010.  These are versions of :math:`\Delta\Sigma` that are less
  sensitive to small-scale structure.
  
Each of these quantities has four versions within the OffsetNFW code.  We'll take the 
:math:`gamma_t` functions as an example.  First, there's the theoretical computation:

>>> gamma_t = model.gamma_theory(radii, mass, concentration, lens_redshift, source_redshift)

Then there's version based on interpolation tables:

>>> gamma_t = model.gamma(radii, mass, concentration, lens_redshift, source_redshift)

This version also accepts a _centering offset radius_--the size of the offset between the observed
halo center and the true halo center.

>>> gamma_t_miscentered = model.gamma(radii, mass, concentration, lens_redshift, source_redshift,
                                      r_mis = miscentering_radius)

(More information on the accuracy of these computations can be found in :doc:`accuracy`.)

Alternatively, if what you want is the _average_ profile over a distribution of centering offsets,
there are two options: one where the distribution can be described as a (2-D) Gaussian, also known
as a Rayleigh distribution:

>>> gamma_t_rayleigh = model.gamma_Rayleigh(radii, mass, concentration, lens_redshift, 
                                            source_redshift, r_mis = miscentering_radius)

and one where the distribution can be described as exponential:

>>> gamma_t_exponential = model.gamma_Rayleigh(radii, mass, concentration, lens_redshift, 
                                               source_redshift, r_mis = miscentering_radius)

For all of the quantities with miscentering, a fraction of the distribution can be described as
perfectly centered instead with the argument `P_cen`:

>>> gamma_t_half_miscentered = model.gamma(radii, mass, concentration, lens_redshift, 
                                           source_redshift, r_mis = miscentering_radius, P_cen=0.5)

And if you would like to integrate over a discretized source redshift distribution, that can be
done as well:

>>> gamma_t_source_integration = model.gamma(radii, mass, concentration, lens_redshift, 
                                             source_redshift, z_source_pdf=source_redshift_pdf)

Versions of these four quantities (the base [interpolated] case, `_theory`, `_Rayleigh`, and
`_exponential`) are available for all lensing quantities.  Please see the :doc:`nfw`(full documentation for
these objects) for more information on their exact call signatures.

Input argument formats
======================

You can think of the input arguments to the lensing quantity objects as coming in three blocks.
The first block is the radius; the second block is the quantities of the lens (mass, concentration,
lens redshift, miscentering radius, probability of correct centroid and [for `Upsilon` only] inner 
radius :math:`r_0`); and the third block is the quantities of the source galaxies, the source 
redshifts and their associated weights (if desired).  The third block only appears for `kappa`, 
`gamma`, and `g` quantities.

_Within each block_, the arguments you pass must either be scalars or all have the same length.  You
can freely mix scalars and arrays, as long as the arrays are all the same length; you should stick
to vectors/1-D arrays, not higher-order arrays, to avoid problems with numpy broadcasting rules.  
For instance, it would be fine to pass a single mass and lens redshift, but a range of lens
concentrations.  Or if you had a catalog of lenses, you could pass the mass column as the mass, the
concentration column as the concentration, and the redshift column as the lens redshift.  

However, there is _no requirement_ that each block be the same shape.  So you could pass a vector
of 10 radii, a mass list of 5 items, and a source redshift list of 7 items.

Any input that is dimensionful can be passed as an `astropy.Quantity`.  If this is not done, the
assumption is made that masses were given in solar masses and any radial quantities (the radius
itself and any miscentering radii) are given in Mpc.

Outputs
=======

OffsetNFW will return a numpy array.  For quantities such as :math:`\Delta\Sigma` that are
dimensionful, the return value will be an astropy.Quantity with appropriate units.

For quantities with no source redshift information, the returned array will be an 
:math:`(m \times n)` array, where `m` is the length of the arrays in the _second_ (lens quantity) 
block and `n` is the length of the array in the _first_ (radial) block.  That is,

>>> len(radii)
6
>>> len(mass)
2
>>> model.deltasigma(radii, mass, concentration, lens_redshift).shape
(6, 2)

The profile from the first lens is thus
>>> result[0]
and the value of the profile at the first radial point for every lens is
>>> result[:, 0]

Any block that contains only scalars will be omitted from the shape of the returned array.  So, for 
example,

>>> model.deltasigma(radii, mass[0], concentration, lens_redshift).shape
(6, )

(rather than `(6, 1)`).  

If an array of source redshifts are given, but no associated weights are given, then the output 
dimensions will be `(l \times m \times n)`, with the length of the source redshift array as the 
first dimension.  In this case, `result[0]` will be all the models for the first source redshift
in the array, `result[0,0]` will be the model for the first source redshift and the first lens, etc.

If a source redshift PDF was also given, then it will be integrated over before the values are
returned; in that case, even if the source redshifts are an array of length :math:`l`, the
returned dimensions will still only be :math:`(m \times n)`.  Of course, the lengths of the two
source-redshift-related arrays must match if they are both arrays!  Note that the PDF will *not* be
renormalized in any way.  It's just going to get multiplied by the arrays and then summed.  
Proper normalization is left to the end user.

Generating interpolation tables
===============================

If you have not yet generated any interpolation tables, it is easy to do so.  Simply make a
NFWModel instance, as above, making sure you've specified the directory where you'd like to save
the tables, and then run:

>>> model.generate()

Timing information can be found in the documentation describing :doc:`accuracy`.  You can choose to 
generate tables only for the use cases you want; see the :method:`offset_nfw.NFWModel.generate`
documentation for the full specification.

Convenience functions
=====================
In addition to the lensing quantities, some useful available methods are:
- :method:`offset_nfw.NFWModel.reference_density`: the reference density at a given redshift, given
  the comoving/physical choice and the matter density/critical density choice made when the 
  `NFWModel` object was created.
- :method:`offset_nfw.NFWModel.scale_radius`: the scale radius of the halo
- :method:`offset_nfw.NFWModel.sigma_to_deltasigma`: given a `sigma` profile sampled at some radii,
  return a deltasigma profile. Note that even for very close spacing in radii, this is likely to be
  unreliable in the innermost order of magnitude or so due to numeric effects.

