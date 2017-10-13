import numpy
import scipy.interpolate
import os
from .utils import reshape, reshape_multisource
try:
    import multiprocessing
    use_multiprocessing = True
except ImportError:
    use_multiprocessing = False
from functools import partial

import astropy.units as u



class NFWModel(object):
    """
    A class that generates offset NFW halo profiles.  The basic purpose of this class is to generate
    internal interpolation tables for the common NFW lensing quantities, but it includes direct
    computation of the non-miscentered versions for completeness.
    
    Initializing a class is easy.  You do need a cosmology object like those created by astropy,
    since we need to know overdensities.  Once you have one:
    >>>  from offset_nfw import NFWModel
    >>>  nfw_model = NFWModel(cosmology)
    
    However, this won't have any internal interpolation tables (unless you've already created them
    in the directory you're working in).  To do that, you pass:
    >>>  nfw_model = NFWModel(generate=True)
    If you want to use tables you generated in another directory, that's easy:
    >>>  nfw_model = NFWModel(dir='nfw_tables')
    Note that setting ``generate=True`` will only generate new internal interpolation tables `if
    those tables do not already exist`.  If you want to `re`generate a table, you should delete
    the table files. They all start `.saved_nfw*` and use the extension `.npy`.
    
    Parameters
    ----------
    cosmology : astropy.cosmology instance
        A cosmology object that can return distances and densities for computing sigma crit and
        rho_m or rho_c.  (Technically, this doesn't have to be an astropy.cosmology instance if it
        has the methods ``angular_diameter_distance``, ``angular_diameter_distance_z1z2``, and 
        ``efunc`` (=H(z)/H0)), plus the attributes ``critical_density0`` and ``Om0``,
    dir : str
        The directory where the saved tables should be stored (will be interpreted through
        ``os.path``). [default: '.']
    rho : str
        Which type of overdensity to use for the halo, 'rho_m' or 'rho_c'. [default: 'rho_m']
    comoving: bool
        Use comoving coordinates (True) or physical coordinates (False). [default: True]
    delta : float
        The overdensity at which the halo mass is defined [default: 200]
    precision : float
        The maximum allowable fractional error, defined for some mass range and concentration TBD
    x_range : tuple
        The min-max range of x (=r/r_s) for the interpolation table. Precision is not guaranteed for 
        values other than the default.  [default: (0.0003, 300)]
    """
    def __init__(self, cosmology, dir='.', rho='rho_m', comoving=True, delta=200,
        precision=0.01, x_range=(0.0003, 300)):

        if not os.path.exists(dir):
            raise RuntimeError("Nonexistent save directory passed to NFWModel")
        self.dir = dir

        if not (hasattr(cosmology, "angular_diameter_distance") and 
                hasattr(cosmology, "angular_diameter_distance_z1z2") and
                hasattr(cosmology, "Om")):
            raise RuntimeError("Must pass working cosmology object to NFWModel")
        self.cosmology = cosmology

        if not rho in ['rho_c', 'rho_m']:
            raise RuntimeError("Only rho_c and rho_m currently implemented")
        self.rho = rho
        
        # Ordinarily I prefer Python duck-typing, but I want to avoid the case where somebody
        # passes "comoving='physical'" and gets comoving coordinates instead because
        # `if 'physical'` evaluates to True!
        if not isinstance(comoving, bool):
            raise RuntimeError("comoving must be True or False")
        self.comoving = comoving
        
        try:
            float(delta)
        except:
            raise RuntimeError("Delta must be a real number")
        if not delta>0:
            raise RuntimeError("Delta<=0 is not physically sensible")
        self.delta = delta
        
        try:
            float(precision)
        except:
            raise RuntimeError("Precision must be a real number")
        if not precision>0:
            raise RuntimeError("Precision must be greater than 0")
        self.precision = precision
        
        if not hasattr(x_range, '__iter__'):
            raise RuntimeError("X range must be a length-2 tuple")
        x_range = numpy.asarray(x_range)
        if numpy.product(x_range.shape)!=2 or len(x_range)!=2:
            raise RuntimeError("X range must be a length-2 tuple")
        try:
            numpy.array(x_range, dtype=float)
        except:
            raise RuntimeError("X range must be composed of real numbers")
        if x_range[0]<=0 or x_range[1]<=0:
            raise RuntimeError("X range must be >=0")
        if x_range[0]>x_range[1]:
            x_range = [x_range[1], x_range[0]]
        self.x_range = x_range
        
        # Useful quantity in scaling profiles
        self._rmod = (3./(4.*numpy.pi)/self.delta)**0.33333333
        
        if hasattr(self.cosmology, 'sigma_crit_inverse'):
            self.sigma_crit_inverse = self.cosmology.sigma_crit_inverse
        else:
            from functools import partial
            from .cosmology import sigma_crit_inverse
            self.sigma_crit_inverse = partial(sigma_crit_inverse, self.cosmology)

    def generate(self, do_sigma=True, do_deltasigma=True, do_rayleigh=False, do_exponential=False):
        self._setupTables()
        if do_rayleigh:
            self._setupRayleighProbabilities()
        if do_exponential:
            self.setupExponentialProbabilities()
        if do_sigma or do_deltasigma:
            self._setupSigma()
            self._buildSigma()
            self._setupMiscenteredSigma()
            if do_rayleigh and do_sigma:
                self._buildRayleighSigma()
            if do_exponential and do_sigma:
                self._buildExponentialSigma()
        if do_deltasigma:
            self._buildMiscenteredSigma()
            self._setupMiscenteredDeltaSigma()
            if do_rayleigh:
                self._buildRayleighDeltaSigma()
            if do_exponential:
                self._buildExponentialDeltaSigma()

    def _setupTables(self):
        self.table_x = numpy.logspace(numpy.log10(self.x_range[0]), numpy.log10(self.x_range[1]), 
                                      num=int(50./self.precision))
        self.x_min = numpy.min(self.table_x)
        self.x_max = numpy.max(self.table_x)
        self.dx = self.table_x[1]/self.table_x[0]
        table_angle = numpy.linspace(0, 2.*numpy.pi, int(2.*numpy.pi/self.precision))
        self.cos_theta_table = numpy.cos(table_angle)[:, numpy.newaxis]


    def _buildSigma(self):
        self._sigma = numpy.zeros_like(self.table_x)
        xm = self.table_x<1
        self._sigma[xm] = self._sigmalt(self.table_x[xm])
        xm = self.table_x==1
        self._sigma[xm] = self._sigmaeq(self.table_x[xm])
        xm = self.table_x>1
        self._sigma[xm] = self._sigmagt(self.table_x[xm])

    def _setupSigma(self):
        self._sigma_table = scipy.interpolate.interp1d(numpy.log(self.table_x), self._sigma,
                                                       fill_value=0, bounds_error=False)

    def _buildMiscenteredSigma(self):
        self._miscentered_sigma = numpy.array([numpy.sum(
            self._sigma_table(
                0.5*numpy.log(
                    self.table_x*self.table_x+tx*tx
                        +2*self.table_x*self.cos_theta_table*tx)),axis=0) for tx in self.table_x])
        # TODO: figure out if you need to trim egregious problems        

    def _setupMiscenteredSigma(self):
        self._miscentered_sigma_table = scipy.interpolate.RegularGridInterpolator(
            (numpy.log(self.table_x), numpy.log(self.table_x)), self._miscentered_sigma)

    def _buildDeltaSigma(self):
        self._deltasigma = numpy.zeros_like(self.table_x)
        xm = self.table_x<1
        self._deltasigma[xm] = self._deltasigmalt(self.table_x[xm])
        xm = self.table_x==1
        self._deltasigma[xm] = self._deltasigmaeq(self.table_x[xm])
        xm = self.table_x>1
        self._deltasigma[xm] = self._deltasigmagt(self.table_x[xm])

    def _setupDeltaSigma(self):
        self._deltasigma_table = scipy.interpolate.interp1d(numpy.log(self.table_x), self._deltasigma,
                                                       fill_value=0, bounds_error=False)

    def _buildMiscenteredDeltaSigma(self):
        self._miscentered_deltasigma = numpy.array([self.sigma_to_deltasigma(self.table_x, ms) for ms in self._miscentered_sigma])

    def _setupMiscenteredDeltaSigma(self):
        self._miscentered_deltasigma_table = scipy.interpolate.RegularGridInterpolator(
            (numpy.log(self.table_x), numpy.log(self.table_x)), self._miscentered_deltasigma)
        
        
    # Per Brainerd and Wright (arXiv:), these are the analytic descriptions of the 
    # NFW lensing profiles.
    def _deltasigmalt(self,x):
        return (8.*numpy.arctanh(numpy.sqrt((1.-x)/(1.+x)))/(x*x*numpy.sqrt(1.-x*x))+
            4./(x*x)*numpy.log(x/2.)-2./(x*x-1.)+
            4.*numpy.arctanh(numpy.sqrt((1.-x)/(1.+x)))/((x*x-1.)*numpy.sqrt(1.-x*x)))
    def _deltasigmaeq(self,x):
        return 10./3.+4.*numpy.log(0.5)
    def _deltasigmagt(self,x):
        return (8.*numpy.arctan(numpy.sqrt((x-1.)/(1.+x)))/(x*x*numpy.sqrt(x*x-1.)) +
            4./(x*x)*numpy.log(x/2.)-2./(x*x-1.)+
            4.*numpy.arctan(numpy.sqrt((x-1.)/(1.+x)))/(pow((x*x-1.),1.5)))
    def _sigmalt(self,x):
        return 2./(x*x-1.)*(1.-2./numpy.sqrt(1.-x*x)*numpy.arctanh(numpy.sqrt((1.-x)/(1.+x))))
    def _sigmaeq(self,x):
        return 2./3.
    def _sigmagt(self,x):
        return 2./(x*x-1.)*(1.-2./numpy.sqrt(x*x-1.)*numpy.arctan(numpy.sqrt((x-1.)/(1.+x))))

    def _filename(self):
        return ''
        
    def sigma_to_deltasigma(self, r, sigma):
        """central_value is default 0; central_value = [something floating-point] will be used; 
        central_value = 'interp' will use the innermost value of sigma.  central_value must have same
        units as sigma, if given explicitly."""
        if hasattr(r, 'unit'):
            r_unit = r.unit
            r = r.value
        else:
            r_unit = 1
        if hasattr(sigma, 'unit'):
            sigma_unit = sigma.unit
            sigma = sigma.value
        else:
            sigma_unit = 1
        sigma_r = 2*numpy.pi*r*sigma
        sum_sigma = scipy.integrate.cumtrapz(sigma_r, r, initial=0)*sigma_unit*r_unit**2
        sum_area = numpy.pi*(r**2-r[0]**2)*r_unit**2
        deltasigma = numpy.zeros_like(sum_sigma)
        # Linearly interpolate central value, which is nan due to sum_area==0
        if len(deltasigma.shape)==1:
            deltasigma[1:] = sum_sigma[1:]/sum_area[1:] - sigma[1:]*sigma_unit        
            deltasigma[0] = 2*deltasigma[1]-deltasigma[2]
        else:
            deltasigma[:,1:] = sum_sigma[:,1:]/sum_area[1:] - sigma[:,1:]*sigma_unit        
            deltasigma[:,0] = 2*deltasigma[:,1]-deltasigma[:,2]
        return deltasigma
        
    def reference_density(self, z):
        """Return the reference density for this halo: that is, critical density for rho_c,
           or matter density for rho_m, properly in comoving or physical."""
        if self.rho=='rho_c':
            dens = self.cosmology.critical_density(z)
            if self.comoving:
                return dens/(1.+z)**3
            else:
                return dens
        else:
            dens = self.cosmology.Om0*self.cosmology.critical_density0
            if self.comoving:
                return dens
            else:
                return dens*(1.+z)**3

    def scale_radius(self, M, c, z):
        """ Return the scale radius in comoving Mpc. """
        c = numpy.asarray(c)
        z = numpy.asarray(z)
        if not isinstance(M, u.Quantity):
            M = (M*u.Msun).to(u.g)
        rs = self._rmod/c*(M/self.reference_density(z))**0.33333333
        return rs.to(u.Mpc**0.99999999).value*u.Mpc  # to deal with fractional powers

    def nfw_norm(self, M, c, z):
        """ Return the normalization for delta sigma and sigma. """
        c = numpy.asarray(c)
        z = numpy.asarray(z)
        if not isinstance(M, u.Quantity):
            M = (numpy.asarray(M)*u.Msun).to(u.g)
        deltac=self.delta/3.*c*c*c/(numpy.log(1.+c)-c/(1.+c))
        rs = self.scale_radius(M, c, z)
        return rs*deltac*self.reference_density(z)

    @reshape
    def deltasigma_theory(self, r, M, c, z):
        """Return an NFW delta sigma from theory.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        z : float or iterable
            The redshift of the halo.  If this is an iterable, all other non-r parameters must be
            either iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of delta sigma at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        rs = self.scale_radius(M, c, z)
        if not isinstance(r, u.Quantity):
            r = r*u.Mpc
        x = numpy.atleast_1d((r/rs).decompose().value)
        
        norm = self.nfw_norm(M, c, z)
        return_vals = numpy.atleast_1d(numpy.zeros_like(x))
        ltmask = x<1
        return_vals[ltmask] = self._deltasigmalt(x[ltmask])
        gtmask = x>1
        return_vals[gtmask] = self._deltasigmagt(x[gtmask])
        eqmask = x==1
        return_vals[eqmask] = self._deltasigmaeq(x[eqmask])
        return_vals = norm*return_vals
        return return_vals
        
    @reshape
    def sigma_theory(self, r, M, c, z):
        """Return an NFW sigma from theory.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of sigma at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        rs = self.scale_radius(M, c, z)
        if not isinstance(r, u.Quantity):
            r = r*u.Mpc
        x = numpy.atleast_1d((r/rs).decompose().value)
        norm = self.nfw_norm(M, c, z)
        return_vals = numpy.atleast_1d(numpy.zeros_like(x))
        ltmask = x<1
        return_vals[ltmask] = self._sigmalt(x[ltmask])
        gtmask = x>1
        return_vals[gtmask] = self._sigmagt(x[gtmask])
        eqmask = x==1
        return_vals[eqmask] = self._sigmaeq(x[eqmask])
        return_vals = norm*return_vals #*= doesn't propagate units
        return return_vals

    @reshape
    def rho_theory(self, r, M, c, z):
        """Return an NFW rho from theory.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of sigma at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        rs = self.scale_radius(M, c, z)
        if not isinstance(r, u.Quantity):
            r *= u.Mpc
        x = numpy.atleast_1d((r/rs).decompose().value)
        norm = self.nfw_norm(M, c, z)/rs
        return norm/(x*(1.+x)**2)

    @reshape        
    def Upsilon_theory(self, r, M, c, z, r0):
        """Return an NFW Upsilon statistic from theory.  
        
        The Upsilon statistics were introduced in Baldauf et al 2010 and Mandelbaum et al 2010 and
        are also called the annular differential surface density (ADSD) statistics.  They are given
        by
        
        ..math:
            \Upsilon(r; r_0) = \Delta\Sigma(r) - \left(\frac{r_0}{r}\right)^2 \Delta\Sigma(r_0)
            
        and remove the dependence on scales below ``r0``.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of Upsilon at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        return (self.deltasigma_theory(r, M, c, z, skip_reformat=True) 
                    - (r0/r)**2*self.deltasigma_theory(r0, M, c, z, skip_reformat=True))

    @reshape_multisource
    def gamma_theory(self, r, M, c, z_lens, z_source, z_source_pdf=None):
        """Return an NFW tangential shear from theory.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of delta sigma at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        deltasigma = self.deltasigma_theory(r, M, c, z_lens, skip_reformat=True)
        sci = self.sigma_crit_inverse(z_lens, z_source)
        return sci*deltasigma

    @reshape_multisource
    def kappa_theory(self, r, M, c, z_lens, z_source, z_source_pdf=None):
        """Return an NFW convergence from theory.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of kappa at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        sigma = self.sigma_theory(r, M, c, z_lens, skip_reformat=True)
        sci = self.sigma_crit_inverse(z_lens, z_source)
        return sci*sigma

    @reshape_multisource
    def g_theory(self, r, M, c, z_lens, z_source, z_source_pdf=None):
        """Return an NFW reduced shear from theory.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of g at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        return (self.gamma_theory(r, M, c, z_lens, z_source, skip_reformat=True)
                 /(1.-self.kappa_theory(r, M, c, z_lens, z_source, skip_reformat=True)))

    @reshape
    def deltasigma(self, r, M, c, r_mis):
        """Return an optionally miscentered NFW delta sigma from an internal interpolation table.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        r_mis : float or iterable
            The distance (in length units) between the reported center of the halo and the actual
            assumed to be in Mpc/h.
            center of the halo.  Whatever definition (comoving/not, etc) you used for your cosmology
            object should be replicated here.  This can be an object with astropy units of length;
            if not it is assumed to be in Mpc/h.  If this is an iterable, all other non-r parameters
            must be either iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of delta sigma at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        raise NotImplementedError("NFWModel currently can't do delta sigmas!")

    @reshape
    def sigma(self, r, M, c, r_mis):
        """Return an optionally miscentered NFW sigma from an internal interpolation table.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        r_mis : float or iterable
            The distance (in length units) between the reported center of the halo and the actual
            assumed to be in Mpc/h.
            center of the halo.  Whatever definition (comoving/not, etc) you used for your cosmology
            object should be replicated here.  This can be an object with astropy units of length;
            if not it is assumed to be in Mpc/h.  If this is an iterable, all other non-r parameters
            must be either iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of sigma at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        raise NotImplementedError("NFWModel currently can't do sigmas!")

    @reshape
    def Upsilon(self, r, M, c, r0, r_mis):
        """Return an optionally miscentered NFW Upsilon statistic from an internal interpolation table.
        
        For details of the Upsilon statistic, see the documentation for :func:`Upsilon_theory`.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        r_mis : float or iterable
            The distance (in length units) between the reported center of the halo and the actual
            assumed to be in Mpc/h.
            center of the halo.  Whatever definition (comoving/not, etc) you used for your cosmology
            object should be replicated here.  This can be an object with astropy units of length;
            if not it is assumed to be in Mpc/h.  If this is an iterable, all other non-r parameters
            must be either iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of delta sigma at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        raise NotImplementedError("NFWModel currently can't do upsilon statistics!")

    @reshape_multisource
    def gamma(self, r, M, c, r_mis, z_lens, z_source, z_source_pdf=None):
        """Return an optionally miscentered NFW tangential shear from an internal interpolation
        table.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        r_mis : float or iterable
            The distance (in length units) between the reported center of the halo and the actual
            assumed to be in Mpc/h.
            center of the halo.  Whatever definition (comoving/not, etc) you used for your cosmology
            object should be replicated here.  This can be an object with astropy units of length;
            if not it is assumed to be in Mpc/h.  If this is an iterable, all other non-r parameters
            must be either iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of gamma at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        raise NotImplementedError("NFWModel currently can't do tangential shear!")

    @reshape_multisource
    def kappa(self, r, M, c, r_mis, z_lens, z_source, z_source_pdf=None):
        """Return an optionally miscentered NFW convergence from an internal interpolation table.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        r_mis : float or iterable
            The distance (in length units) between the reported center of the halo and the actual
            assumed to be in Mpc/h.
            center of the halo.  Whatever definition (comoving/not, etc) you used for your cosmology
            object should be replicated here.  This can be an object with astropy units of length;
            if not it is assumed to be in Mpc/h.  If this is an iterable, all other non-r parameters
            must be either iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of kappa at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        raise NotImplementedError("NFWModel currently can't do convergence!")

    @reshape_multisource       
    def g(self, r, M, c, r_mis, z_lens, z_source, z_source_pdf=None):
        """Return an optionally miscentered NFW reduced shear from an internal interpolation table.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        r_mis : float or iterable
            The distance (in length units) between the reported center of the halo and the actual
            assumed to be in Mpc/h.
            center of the halo.  Whatever definition (comoving/not, etc) you used for your cosmology
            object should be replicated here.  This can be an object with astropy units of length;
            if not it is assumed to be in Mpc/h.  If this is an iterable, all other non-r parameters
            must be either iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of g at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        raise NotImplementedError("NFWModel currently can't do reduced shear!")

    @reshape
    def deltasigma_Rayleigh(self, r, M, c, r_mis, P_cen):
        """Return an NFW delta sigma from an internal interpolation table, assuming that the
        miscentering takes the form of a Rayleigh (2d Gaussian) distribution plus a delta function:
        fraction `0<P_cen<1` of the halos have correct centers, while the ones which are miscentered
        have a Rayleigh distribution with width r_mis.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        r_mis : float or iterable
            The distance (in length units) between the reported center of the halo and the actual
            assumed to be in Mpc/h.
            center of the halo.  Whatever definition (comoving/not, etc) you used for your cosmology
            object should be replicated here.  This can be an object with astropy units of length;
            if not it is assumed to be in Mpc/h.  If this is an iterable, all other non-r parameters
            must be either iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of delta sigma at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        raise NotImplementedError("NFWModel currently can't do delta sigmas with Rayleigh "+
                "distributions!")

    @reshape
    def sigma_Rayleigh(self, r, M, c, r_mis, P_cen):
        """Return an NFW sigma from an internal interpolation table, assuming that the
        miscentering takes the form of a Rayleigh (2d Gaussian) distribution plus a delta function:
        fraction `0<P_cen<1` of the halos have correct centers, while the ones which are miscentered
        have a Rayleigh distribution with width r_mis.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        r_mis : float or iterable
            The distance (in length units) between the reported center of the halo and the actual
            assumed to be in Mpc/h.
            center of the halo.  Whatever definition (comoving/not, etc) you used for your cosmology
            object should be replicated here.  This can be an object with astropy units of length;
            if not it is assumed to be in Mpc/h.  If this is an iterable, all other non-r parameters
            must be either iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of sigma at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        raise NotImplementedError("NFWModel currently can't do sigmas with Rayleigh distributions!")

    @reshape
    def Upsilon_Rayleigh(self, r, M, c, r0, r_mis, P_cen):
        """Return an NFW Upsilon statistic from an internal interpolation table, assuming that the
        miscentering takes the form of a Rayleigh (2d Gaussian) distribution plus a delta function:
        fraction `0<P_cen<1` of the halos have correct centers, while the ones which are miscentered
        have a Rayleigh distribution with width r_mis.
        
        For details of the Upsilon statistic, see the documentation for :func:`Upsilon_theory`.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        r_mis : float or iterable
            The distance (in length units) between the reported center of the halo and the actual
            assumed to be in Mpc/h.
            center of the halo.  Whatever definition (comoving/not, etc) you used for your cosmology
            object should be replicated here.  This can be an object with astropy units of length;
            if not it is assumed to be in Mpc/h.  If this is an iterable, all other non-r parameters
            must be either iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of Upsilon at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        raise NotImplementedError("NFWModel currently can't do upsilon statistics with Rayleigh "+
                "distributions!")

    @reshape_multisource
    def gamma_Rayleigh(self, r, M, c, r_mis, P_cen, z_lens, z_source, z_source_pdf=None):
        """Return an NFW tangential shear from an internal interpolation table, assuming that the
        miscentering takes the form of a Rayleigh (2d Gaussian) distribution plus a delta function:
        fraction `0<P_cen<1` of the halos have correct centers, while the ones which are miscentered
        have a Rayleigh distribution with width r_mis.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        r_mis : float or iterable
            The distance (in length units) between the reported center of the halo and the actual
            assumed to be in Mpc/h.
            center of the halo.  Whatever definition (comoving/not, etc) you used for your cosmology
            object should be replicated here.  This can be an object with astropy units of length;
            if not it is assumed to be in Mpc/h.  If this is an iterable, all other non-r parameters
            must be either iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of gamma at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        raise NotImplementedError("NFWModel currently can't do tangential shear with Rayleigh "+
                "distributions!")

    @reshape_multisource    
    def kappa_Rayleigh(self, r, M, c, r_mis, P_cen, z_lens, z_source, z_source_pdf=None):
        """Return an NFW convergence from an internal interpolation table, assuming that the
        miscentering takes the form of a Rayleigh (2d Gaussian) distribution plus a delta function:
        fraction `0<P_cen<1` of the halos have correct centers, while the ones which are miscentered
        have a Rayleigh distribution with width r_mis.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        r_mis : float or iterable
            The distance (in length units) between the reported center of the halo and the actual
            assumed to be in Mpc/h.
            center of the halo.  Whatever definition (comoving/not, etc) you used for your cosmology
            object should be replicated here.  This can be an object with astropy units of length;
            if not it is assumed to be in Mpc/h.  If this is an iterable, all other non-r parameters
            must be either iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of kappa at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        raise NotImplementedError("NFWModel currently can't do convergence with Rayleigh "+
                "distributions!")
    
    @reshape_multisource
    def g_Rayleigh(self, r, M, c, r_mis, P_cen, z_lens, z_source, z_source_pdf=None):
        """Return an NFW reduced shear from an internal interpolation table, assuming that the
        miscentering takes the form of a Rayleigh (2d Gaussian) distribution plus a delta function:
        fraction `0<P_cen<1` of the halos have correct centers, while the ones which are miscentered
        have a Rayleigh distribution with width r_mis.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        r_mis : float or iterable
            The distance (in length units) between the reported center of the halo and the actual
            assumed to be in Mpc/h.
            center of the halo.  Whatever definition (comoving/not, etc) you used for your cosmology
            object should be replicated here.  This can be an object with astropy units of length;
            if not it is assumed to be in Mpc/h.  If this is an iterable, all other non-r parameters
            must be either iterables with the same length or floats.
        
        Returns
        -------
        float or numpy.ndarray
            Returns the value of g at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        raise NotImplementedError("NFWModel currently can't do reduced shear with Rayleigh "+
                "distributions!")
