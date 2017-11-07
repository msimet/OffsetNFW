import numpy
import numpy.testing
import astropy.cosmology
import astropy.units as u
import os
import glob
try:
    import offset_nfw
except ImportError:
    import sys
    sys.path.append('..')
    import offset_nfw
    

# A "cosmology" object that passes initialization tests.
class fake_cosmo(object):
    critical_density0 = 1
    Om0 = 1
    def critical_density(self, x):
        return 1
    def angular_diameter_distance(self, x):
        return 1
    def angular_diameter_distance_z1z2(self, x, y):
        return 1
    def Om(self, x):
        return 1

m_c_z_test_list = [(1E14, 4, 0.2), (1E13, 4, 0.2), (1E15, 4, 0.2),
                   (1E14, 2, 0.2), (1E14, 6, 0.2), 
                   (1E14, 4, 0.05), (1E14, 4, 0.5), (1E14, 4, 4)]
m_c_z_multi_test_list = [([1E14, 1E15], 4, 0.2), 
                         (1E14, [2,4,6], 0.2), 
                         (1E14, 4, [0.2,0.5])]
cosmo = astropy.cosmology.FlatLambdaCDM(H0=100, Om0=0.3)
        

def test_object_creation():
    # Need a cosmology object
    numpy.testing.assert_raises(TypeError, offset_nfw.NFWModel)
    # Need a REAL cosmology object
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, None) 
    cosmology_obj = fake_cosmo()
    offset_nfw.NFWModel(cosmology_obj)  # This should pass
    # Existing directory
    offset_nfw.NFWModel(cosmology_obj, dir='.')
    # Non-existing directory
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, cosmology_obj, dir='_random_dir')
    # Wrong rho type
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, cosmology_obj, rho='rho_dm')
    # Nonsensical delta
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, cosmology_obj, delta=-200)
    # Non-working ranges
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, cosmology_obj, x_range=3)
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, cosmology_obj, x_range=(3,4,5))
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, cosmology_obj, x_range=('a', 'b'))
    # Should work
    offset_nfw.NFWModel(cosmology_obj, x_range=[3,4])
    # Non-working ranges

    obj = offset_nfw.NFWModel(cosmology_obj, '.', 'rho_m', delta=150, precision=0.02, x_range=(0.1,2), 
                       comoving=False)
    numpy.testing.assert_equal(obj.cosmology, cosmology_obj)
    numpy.testing.assert_equal(obj.dir, '.')
    numpy.testing.assert_equal(obj.rho, 'rho_m')
    numpy.testing.assert_equal(obj.delta, 150)
    numpy.testing.assert_equal(obj.precision, 0.02)
    numpy.testing.assert_equal(obj.x_range, (0.1,2))
    numpy.testing.assert_equal(obj.comoving, False)
    
    # Should work
    offset_nfw.NFWModel(astropy.cosmology.FlatLambdaCDM(H0=100, Om0=0.3))

def test_scale_radii():
    """ Test scale radius measurement. """
    # Test against some precomputed values
    nfw_1 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c')
    numpy.testing.assert_allclose(nfw_1.scale_radius(1E14, 4, 0.2).to(u.Mpc).value, 0.2120377818122246)
    numpy.testing.assert_allclose(nfw_1.scale_radius(1E15, 3, 0.2).to(u.Mpc).value, 0.609095398969911)
    numpy.testing.assert_allclose(nfw_1.scale_radius(1E13, 5, 0.2).to(u.Mpc).value, 0.07873537663340793)
    numpy.testing.assert_allclose(nfw_1.scale_radius(1E14, 4, 0.1).to(u.Mpc).value, 0.20114937491773577)
    numpy.testing.assert_allclose(nfw_1.scale_radius(1E14, 4.5, 0.3).to(u.Mpc).value, 0.1968790019866928)
    nfw_2 = offset_nfw.NFWModel(cosmo, delta=150, rho='rho_c')
    numpy.testing.assert_allclose(nfw_2.scale_radius(1E14, 4, 0.2).to(u.Mpc).value, 0.23337777629652395)
    numpy.testing.assert_allclose(nfw_2.scale_radius(1E15, 3, 0.2).to(u.Mpc).value, 0.6703962310354946)
    numpy.testing.assert_allclose(nfw_2.scale_radius(1E13, 5, 0.2).to(u.Mpc).value, 0.08665949510284233)
    numpy.testing.assert_allclose(nfw_2.scale_radius(1E14, 4, 0.1).to(u.Mpc).value, 0.22139353383402788)
    numpy.testing.assert_allclose(nfw_2.scale_radius(1E14, 4.5, 0.3).to(u.Mpc).value, 0.21669338025721746)
    nfw_3 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_m')
    # These were computed using separate code, hence almost_equal instead of equal
    numpy.testing.assert_almost_equal(nfw_3.scale_radius(1E14, 4, 0.2).to(u.Mpc).value, 0.281924022285, decimal=4)
    numpy.testing.assert_almost_equal(nfw_3.scale_radius(1E15, 3, 0.2).to(u.Mpc).value, 0.809849191419, decimal=4)
    numpy.testing.assert_almost_equal(nfw_3.scale_radius(1E13, 5, 0.2).to(u.Mpc).value, 0.104686031501, decimal=4)
    numpy.testing.assert_almost_equal(nfw_3.scale_radius(1E14, 4, 0.1).to(u.Mpc).value, 0.281924022285, decimal=4)
    numpy.testing.assert_almost_equal(nfw_3.scale_radius(1E14, 4.5, 0.3).to(u.Mpc).value, 0.25059913092, decimal=4)
    nfw_4 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_m', comoving=False)
    numpy.testing.assert_allclose(nfw_3.scale_radius(1E14, 4, 0.2).to(u.Mpc).value, 
                               1.2*nfw_4.scale_radius(1E14, 4, 0.2).to(u.Mpc).value)
    numpy.testing.assert_allclose(nfw_3.scale_radius(1E15, 3, 0.2).to(u.Mpc).value, 
                               1.2*nfw_4.scale_radius(1E15, 3, 0.2).to(u.Mpc).value)
    numpy.testing.assert_allclose(nfw_3.scale_radius(1E13, 5, 0.2).to(u.Mpc).value, 
                               1.2*nfw_4.scale_radius(1E13, 5, 0.2).to(u.Mpc).value)
    numpy.testing.assert_allclose(nfw_3.scale_radius(1E14, 4, 0.1).to(u.Mpc).value, 
                               1.1*nfw_4.scale_radius(1E14, 4, 0.1).to(u.Mpc).value)
    numpy.testing.assert_allclose(nfw_3.scale_radius(1E14, 4.5, 0.3).to(u.Mpc).value, 
                               1.3*nfw_4.scale_radius(1E14, 4.5, 0.3).to(u.Mpc).value)
    nfw_5 = offset_nfw.NFWModel(cosmo, delta=150, rho='rho_c', comoving=False)
    numpy.testing.assert_allclose(nfw_2.scale_radius(1E14, 4, 0.2).to(u.Mpc).value, 
                               1.2*nfw_5.scale_radius(1E14, 4, 0.2).to(u.Mpc).value)
    numpy.testing.assert_allclose(nfw_2.scale_radius(1E15, 3, 0.2).to(u.Mpc).value, 
                               1.2*nfw_5.scale_radius(1E15, 3, 0.2).to(u.Mpc).value)
    numpy.testing.assert_allclose(nfw_2.scale_radius(1E13, 5, 0.2).to(u.Mpc).value, 
                               1.2*nfw_5.scale_radius(1E13, 5, 0.2).to(u.Mpc).value)
    numpy.testing.assert_allclose(nfw_2.scale_radius(1E14, 4, 0.1).to(u.Mpc).value, 
                               1.1*nfw_5.scale_radius(1E14, 4, 0.1).to(u.Mpc).value)
    numpy.testing.assert_allclose(nfw_2.scale_radius(1E14, 4.5, 0.3).to(u.Mpc).value, 
                               1.3*nfw_5.scale_radius(1E14, 4.5, 0.3).to(u.Mpc).value)
    nfw_6 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c', comoving=False)
    
    try:
        import galsim.nfw_halo
        # There is a hard-coded constant in galsim.nfw_halo that is 3 decimals, so we cannot go
        # more precise than that
        for m, c, z in m_c_z_test_list:
            nfw_comp = galsim.nfw_halo.NFWHalo(m, c, z, omega_m=cosmo.Om0)
            numpy.testing.assert_almost_equal(nfw_6.scale_radius(m, c, z).to(u.Mpc).value, nfw_comp.rs, decimal=3)
    except ImportError:
        pass
    
def test_against_colossus():
        try:
            import colossus.Cosmology, colossus.HaloDensityProfile
            params = {'flat': True, 'H0': 100, 'Om0': 0.3, 'Ob0': 0.043, 'sigma8': 0.8, 'ns': 0.97}
            colossus.Cosmology.setCosmology('myCosmo', params)
            colossus_nfw_1 = colossus.HaloDensityProfile.NFWProfile(M=1E14, c=4, z=0.2, mdef='200m')
            nfw_1 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_m', comoving=False)
            numpy.testing.assert_almost_equal(
                0.001*colossus_nfw_1.RDelta(z=0.2, mdef='200m')/4, 
                nfw_1.scale_radius(1.E14, 4, 0.2).to(u.Mpc).value, decimal=4)
            colossus_nfw_1 = colossus.HaloDensityProfile.NFWProfile(M=1E15, c=4, z=0.2, mdef='200m')
            numpy.testing.assert_almost_equal(
                0.001*colossus_nfw_1.RDelta(z=0.2, mdef='200m')/4, 
                nfw_1.scale_radius(1.E15, 4, 0.2).to(u.Mpc).value, decimal=4)
            colossus_nfw_1 = colossus.HaloDensityProfile.NFWProfile(M=1E13, c=3.5, z=0.2, mdef='200m')
            numpy.testing.assert_almost_equal(
                0.001*colossus_nfw_1.RDelta(z=0.2, mdef='200m')/3.5, 
                nfw_1.scale_radius(1.E13, 3.5, 0.2).to(u.Mpc).value, decimal=4)
            colossus_nfw_1 = colossus.HaloDensityProfile.NFWProfile(M=1E14, c=4, z=0.4, mdef='200m')
            numpy.testing.assert_almost_equal(
                0.001*colossus_nfw_1.RDelta(z=0.4, mdef='200m')/4, 
                nfw_1.scale_radius(1.E14, 4, 0.4).to(u.Mpc).value, decimal=4)
            colossus_nfw_1 = colossus.HaloDensityProfile.NFWProfile(M=1E14, c=4, z=0.4, mdef='180m')
            nfw_2 = offset_nfw.NFWModel(cosmo, delta=180, rho='rho_m', comoving=False)
            numpy.testing.assert_almost_equal(
                0.001*colossus_nfw_1.RDelta(z=0.4, mdef='180m')/4, 
                nfw_2.scale_radius(1.E14, 4, 0.4).to(u.Mpc).value, decimal=4)
            
            colossus_nfw_1 = colossus.HaloDensityProfile.NFWProfile(M=1E14, c=4, z=0.2, mdef='200c')
            nfw_1 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c', comoving=False)
            numpy.testing.assert_almost_equal(
                0.001*colossus_nfw_1.RDelta(z=0.2, mdef='200c')/4, 
                nfw_1.scale_radius(1.E14, 4, 0.2).to(u.Mpc).value, decimal=4)
            colossus_nfw_1 = colossus.HaloDensityProfile.NFWProfile(M=1E15, c=4, z=0.2, mdef='200c')
            numpy.testing.assert_almost_equal(
                0.001*colossus_nfw_1.RDelta(z=0.2, mdef='200c')/4, 
                nfw_1.scale_radius(1.E15, 4, 0.2).to(u.Mpc).value, decimal=4)
            colossus_nfw_1 = colossus.HaloDensityProfile.NFWProfile(M=1E13, c=3.5, z=0.2, mdef='200c')
            numpy.testing.assert_almost_equal(
                0.001*colossus_nfw_1.RDelta(z=0.2, mdef='200c')/3.5, 
                nfw_1.scale_radius(1.E13, 3.5, 0.2).to(u.Mpc).value, decimal=4)
            colossus_nfw_1 = colossus.HaloDensityProfile.NFWProfile(M=1E14, c=4, z=0.4, mdef='200c')
            numpy.testing.assert_almost_equal(
                0.001*colossus_nfw_1.RDelta(z=0.4, mdef='200c')/4, 
                nfw_1.scale_radius(1.E14, 4, 0.4).to(u.Mpc).value, decimal=4)
            colossus_nfw_1 = colossus.HaloDensityProfile.NFWProfile(M=1E14, c=4, z=0.4, mdef='180c')
            nfw_2 = offset_nfw.NFWModel(cosmo, delta=180, rho='rho_c', comoving=False)
            numpy.testing.assert_almost_equal(
                0.001*colossus_nfw_1.RDelta(z=0.4, mdef='180c')/4, 
                nfw_2.scale_radius(1.E14, 4, 0.4).to(u.Mpc).value, decimal=4)
        except ImportError:
            pass
    
def test_against_galsim_theory():
    """ Test against the GalSim implementation of NFWs. """
    try:
        import galsim
        nfw_1 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c', comoving=False)
        
        z_source = 4.95
        
        radbins = numpy.exp(numpy.linspace(numpy.log(0.01), numpy.log(20), num=100))
        for m, c, z in [(1E14, 4, 0.2), (1E13, 4, 0.2), (1E15, 4, 0.2),
                        (1E14, 2, 0.2), (1E14, 6, 0.2), 
                        (1E14, 4, 0.05), (1E14, 4, 0.5), (1E14, 4, 4)]:
            galsim_nfw = galsim.NFWHalo(m, c, z, omega_m = cosmo.Om0)
            angular_pos_x = radbins/cosmo.angular_diameter_distance(z)*206265
            angular_pos_y = numpy.zeros_like(angular_pos_x)
            # want tangential shear; galsim gives us 2-component, but all along x-axis, so just use
            # first component with negative sign
            nfw_1.gamma_theory(radbins, m, c, z, z_source)
            numpy.testing.assert_almost_equal(-galsim_nfw.getShear((angular_pos_x, angular_pos_y), z_source, reduced=False)[0],
                                       nfw_1.gamma_theory(radbins, m, c, z, z_source), decimal=3)
            numpy.testing.assert_almost_equal(galsim_nfw.getConvergence((angular_pos_x, angular_pos_y), z_source), 
                                       nfw_1.kappa_theory(radbins, m, c, z, z_source), decimal=3)
            # Really, we should test reduced shear too. However, the combo of the fact that
            # there's a large scale dependence & we disagree most at large radii means both
            # assert_almost_equal and assert_allclose fail for some range of radii; therefore,
            # we can't pass the test, although the individual pieces of reduced shear do pass.
        
    except ImportError:
        import warnings
        warnings.warn("Could not test against GalSim -- import failure")
        return True
    
def test_against_clusterlensing_theory():
    """ Test against the cluster-lensing implementation of NFWs. """
    try:
        import clusterlensing
    except ImportError:
        import warnings
        warnings.warn("Could not test against cluster-lensing -- import failure")

def test_sigma_to_deltasigma_theory(plot=False):
    """ Test that the numerical sigma -> deltasigma produces the theoretical DS. """
    radbins = numpy.exp(numpy.linspace(numpy.log(0.001), numpy.log(100), num=500))
    nfw_1 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c')
    for m, c, z in m_c_z_test_list:
        ds = nfw_1.deltasigma_theory(radbins, m, c, z)
        sig = nfw_1.sigma_theory(radbins, m, c, z)
        ds_from_sigma = nfw_1.sigma_to_deltasigma(radbins, sig)
        n_to_keep=int(len(radbins)*0.6)
        numpy.testing.assert_almost_equal(ds.value[-n_to_keep:], ds_from_sigma.value[-n_to_keep:], decimal=3)
        numpy.testing.assert_equal(ds.unit, ds_from_sigma.unit)
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(radbins, ds_from_sigma/ds, label="ds")
    #        plt.plot(radbins, ds_from_sigma, label="ds from sigma")
            plt.xscale('log')
            plt.ylim((0., 2))
            plt.savefig('test.png')
    for m, c, z in m_c_z_multi_test_list:
        ds = nfw_1.deltasigma_theory(radbins, m, c, z)
        sig = nfw_1.sigma_theory(radbins, m, c, z)
        ds_from_sigma = nfw_1.sigma_to_deltasigma(radbins, sig)
        n_to_keep=int(len(radbins)*0.6)
        numpy.testing.assert_almost_equal(ds.value[:,-n_to_keep:], ds_from_sigma.value[:,-n_to_keep:], decimal=3)
        numpy.testing.assert_equal(ds.unit, ds_from_sigma.unit)
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(radbins, ds_from_sigma[0]/ds[0], label="ds")
            plt.plot(radbins, ds_from_sigma[1]/ds[1], label="ds")
    #        plt.plot(radbins, ds_from_sigma, label="ds from sigma")
            plt.xscale('log')
            plt.ylim((0., 2))
            plt.savefig('test.png')
    #TODO: do again, miscentered
        
def test_z_ratios_theory():
    """ Test that the theoretical shear changes properly with redshift"""
    nfw_1 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c')
    base = nfw_1.gamma_theory(1., 1.E14, 4, 0.1, 0.15)
    new_z = numpy.linspace(0.15, 1.1, num=20)
    new_gamma = nfw_1.gamma_theory(1, 1.E14, 4, 0.1, new_z)
    new_gamma /= base
    numpy.testing.assert_allclose(new_gamma, cosmo.angular_diameter_distance_z1z2(0.1, new_z)/cosmo.angular_diameter_distance_z1z2(0.1, 0.15)*cosmo.angular_diameter_distance(0.15)/cosmo.angular_diameter_distance(new_z))

    base = nfw_1.kappa_theory(1., 1.E14, 4, 0.1, 0.15)
    new_sigma = nfw_1.kappa_theory(1, 1.E14, 4, 0.1, new_z)
    new_sigma /= base
    numpy.testing.assert_allclose(new_sigma, cosmo.angular_diameter_distance_z1z2(0.1, new_z)/cosmo.angular_diameter_distance_z1z2(0.1, 0.15)*cosmo.angular_diameter_distance(0.15)/cosmo.angular_diameter_distance(new_z))
    
    #TODO: do again, miscentered
    
def test_g():
    """ Test that the theoretical returned g is the appropriate reduced shear """
    radbins = numpy.exp(numpy.linspace(numpy.log(0.001), numpy.log(100), num=500))
    nfw_1 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c')
    z_source = 4.95
    for m, c, z in m_c_z_test_list+m_c_z_multi_test_list:
        numpy.testing.assert_allclose(nfw_1.g_theory(radbins, m, c, z, z_source), 
                                   nfw_1.gamma_theory(radbins, m, c, z, z_source)
                                  /(1-nfw_1.kappa_theory(radbins, m, c, z, z_source)))
    
def test_Upsilon():
    """ Test that the theoretical Upsilon is the appropriate value. """
    radbins = numpy.exp(numpy.linspace(numpy.log(0.001), numpy.log(100), num=500))
    nfw_1 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c')
    for m, c, z in m_c_z_test_list:
        for r in radbins[100:400:4]:
            numpy.testing.assert_allclose(nfw_1.Upsilon_theory(radbins, m, c, z, r).value,
                                       nfw_1.deltasigma_theory(radbins, m, c, z).value 
                                       - (r/radbins)**2*nfw_1.deltasigma_theory(r, m, c, z).value)
    for m, c, z in m_c_z_multi_test_list:
        for r in radbins[100:400:4]:
            numpy.testing.assert_allclose(nfw_1.Upsilon_theory(radbins, m, c, z, r).value,
                       nfw_1.deltasigma_theory(radbins, m, c, z).value 
                       - (r/radbins)**2*nfw_1.deltasigma_theory(r, m, c, z).value[:, numpy.newaxis])

def test_ordering():
    """ Test that the axes are set up properly for multidimensional inputs."""
    radbins = numpy.exp(numpy.linspace(numpy.log(0.001), numpy.log(100), num=10))
    nfw_1 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c')
    m, c, z = m_c_z_test_list[0]
    zs = [z+0.1, z+0.1, z+1]
    
    base_result = nfw_1.deltasigma_theory(radbins, m, c, z)
    comp_m = nfw_1.deltasigma_theory(radbins, [m,m], c, z)
    numpy.testing.assert_equal(comp_m[0], comp_m[1])
    numpy.testing.assert_equal(base_result, comp_m[0])
    comp_c = nfw_1.deltasigma_theory(radbins, m, [c,c], z)
    numpy.testing.assert_equal(comp_c[0], comp_c[1])
    numpy.testing.assert_equal(base_result, comp_c[0])
    comp_z = nfw_1.deltasigma_theory(radbins, m, c, [z,z])
    numpy.testing.assert_equal(comp_z[0], comp_z[1])
    numpy.testing.assert_equal(base_result, comp_z[0])
    
    sub_base_result = nfw_1.g_theory(radbins, m, c, z, zs[0])
    base_result = nfw_1.g_theory(radbins, m, c, z, zs)
    numpy.testing.assert_equal(base_result[0], sub_base_result)
    numpy.testing.assert_equal(base_result[0], base_result[1])
    # There's no "assert not equal", so let's try this
    numpy.testing.assert_raises(AssertionError,
        numpy.testing.assert_equal, base_result[0], base_result[2])
    comp_m = nfw_1.g_theory(radbins, [m,m], c, z, zs)
    numpy.testing.assert_equal(comp_m[:,0], comp_m[:,1])
    numpy.testing.assert_equal(base_result, comp_m[:,0])
    numpy.testing.assert_equal(comp_m[0,0], comp_m[0,1])
    numpy.testing.assert_equal(comp_m[1,0], comp_m[1,1])
    numpy.testing.assert_equal(comp_m[2,0], comp_m[2,1])
    numpy.testing.assert_equal(sub_base_result, comp_m[0,0])
    numpy.testing.assert_raises(AssertionError,
        numpy.testing.assert_equal, comp_m[1,0], comp_m[2,0])
    comp_c = nfw_1.g_theory(radbins, m, [c,c], z, zs)
    numpy.testing.assert_equal(comp_c[:,0], comp_c[:,1])
    numpy.testing.assert_equal(base_result, comp_c[:,0])
    numpy.testing.assert_equal(comp_c[0,0], comp_c[0,1])
    numpy.testing.assert_equal(comp_c[1,0], comp_c[1,1])
    numpy.testing.assert_equal(comp_c[2,0], comp_c[2,1])
    numpy.testing.assert_equal(sub_base_result, comp_c[0,0])
    numpy.testing.assert_raises(AssertionError,
        numpy.testing.assert_equal, comp_c[1,0], comp_c[2,0])
    comp_z = nfw_1.g_theory(radbins, m, c, [z,z], zs)
    numpy.testing.assert_equal(comp_z[:,0], comp_z[:,1])
    numpy.testing.assert_equal(base_result, comp_z[:,0])
    numpy.testing.assert_equal(comp_z[0,0], comp_z[0,1])
    numpy.testing.assert_equal(comp_z[1,0], comp_z[1,1])
    numpy.testing.assert_equal(comp_z[2,0], comp_z[2,1])
    numpy.testing.assert_equal(sub_base_result, comp_z[0,0])
    numpy.testing.assert_raises(AssertionError,
        numpy.testing.assert_equal, comp_z[1,0], comp_z[2,0])


    
def test_setup_table():
    """ Generate a small interpolation table so we can test its outputs. """
    cosmology_obj = fake_cosmo()
    for xr in [(0.1, 0.2), (0.57, 1.83), (5,7)]:
        nfw_halo = offset_nfw.NFWModel(cosmology_obj, x_range=xr)
        nfw_halo._buildTables()
        numpy.testing.assert_array_almost_equal_nulp(nfw_halo.table_x[0], xr[0])
        numpy.testing.assert_array_almost_equal_nulp(nfw_halo.table_x[-1], xr[1])
        numpy.testing.assert_array_almost_equal_nulp(nfw_halo.table_x[0], nfw_halo.x_min)
        numpy.testing.assert_array_almost_equal_nulp(nfw_halo.table_x[-1], nfw_halo.x_max)
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, cosmology_obj, x_range=[-1,1])
    nfw_halo = offset_nfw.NFWModel(cosmology_obj, x_range=[1.0,2.0])
    nfw_halo._buildTables()
    costheta = nfw_halo.cos_theta_table.flatten()
    numpy.testing.assert_equal(costheta[0], 1.)
    numpy.testing.assert_approx_equal(costheta[-1], 1., significant=4)
    numpy.testing.assert_approx_equal(costheta[len(costheta)/2], -1., significant=4)

def test_build_sigma():
    # Just test that it runs--we'll test the real values against sigma_theory later.
    cosmology_obj = fake_cosmo()
    for xr in [(0.1,0.99), (1.01,2.0), (1.0, 2.0), (0.1,2.0)]:
        nfw_halo = offset_nfw.NFWModel(cosmology_obj, x_range=xr)
        nfw_halo._buildTables()
        nfw_halo._buildSigma(save=False)
        numpy.testing.assert_equal(nfw_halo._sigma.shape, (len(nfw_halo.table_x),)) 
        numpy.testing.assert_array_less(0, nfw_halo._sigma)
        nfw_halo._setupSigma()
        for i in range(0,200,20):
            mean_x = numpy.sqrt(nfw_halo.table_x[i]*nfw_halo.table_x[i+1])
            mean_sig = 0.5*(nfw_halo._sigma[i]+nfw_halo._sigma[i+1])
            numpy.testing.assert_approx_equal(nfw_halo._sigma_table(numpy.log(mean_x)), mean_sig)

def test_build_miscentered_sigma():
    cosmology_obj = fake_cosmo()
    for xr in [(0.1,0.99), (1.01,2.0), (1.0, 2.0), (0.1,2.0)]:
        nfw_halo = offset_nfw.NFWModel(cosmology_obj, x_range=xr, precision=1)
        nfw_halo._buildTables()
        nfw_halo._buildSigma(save=False)
        nfw_halo._setupSigma()
        nfw_halo._buildMiscenteredSigma(save=False)
        n = len(nfw_halo.table_x)
        numpy.testing.assert_equal(nfw_halo._miscentered_sigma.shape, (n,n)) 
        nfw_halo._setupMiscenteredSigma()
        for i in range(0,n-1,n/10):
            for j in range(0,n-1,n/10):
                mean_x = numpy.sqrt(nfw_halo.table_x[i]*nfw_halo.table_x[i+1])
                mean_sig = 0.5*(nfw_halo._miscentered_sigma[i,j]+nfw_halo._miscentered_sigma[i+1,j])
                numpy.testing.assert_approx_equal(nfw_halo._miscentered_sigma_table((numpy.log(mean_x), numpy.log(nfw_halo.table_x[j]))), mean_sig)
                numpy.testing.assert_approx_equal(nfw_halo._miscentered_sigma[i][j],
                                                  nfw_halo._miscentered_sigma[j][i])

def test_build_deltasigma():
    # Just test that it runs--we'll test the real values against sigma_theory later.
    cosmology_obj = fake_cosmo()
    for xr in [(0.1,0.99), (1.01,2.0), (1.0, 2.0), (0.1,2.0)]:
        nfw_halo = offset_nfw.NFWModel(cosmology_obj, x_range=xr)
        nfw_halo._buildTables()
        nfw_halo._buildDeltaSigma(save=False)
        numpy.testing.assert_equal(nfw_halo._deltasigma.shape, (len(nfw_halo.table_x),)) 
        numpy.testing.assert_array_less(0, nfw_halo._deltasigma)
        nfw_halo._setupDeltaSigma()
        for i in range(0,200,20):
            mean_x = numpy.sqrt(nfw_halo.table_x[i]*nfw_halo.table_x[i+1])
            mean_ds = 0.5*(nfw_halo._deltasigma[i]+nfw_halo._deltasigma[i+1])
            numpy.testing.assert_approx_equal(nfw_halo._deltasigma_table(numpy.log(mean_x)), mean_ds)


def test_build_miscentered_deltasigma():
    cosmology_obj = fake_cosmo()
    for xr in [(0.1,0.99), (1.01,2.0), (1.0, 2.0), (0.1,2.0)]:
        nfw_halo = offset_nfw.NFWModel(cosmology_obj, x_range=xr, precision=1)
        nfw_halo._buildTables()
        nfw_halo._buildSigma(save=False)
        nfw_halo._setupSigma()
        nfw_halo._buildMiscenteredSigma(save=False)
        nfw_halo._buildMiscenteredDeltaSigma(save=False)
        n = len(nfw_halo.table_x)
        numpy.testing.assert_equal(nfw_halo._miscentered_deltasigma.shape, (n,n)) 
        nfw_halo._setupMiscenteredDeltaSigma()
        for i in range(0,n-1,n/10):
            for j in range(0,n-1,n/10):
                mean_x = numpy.sqrt(nfw_halo.table_x[i]*nfw_halo.table_x[i+1])
                mean_ds = 0.5*(nfw_halo._miscentered_deltasigma[i,j]+nfw_halo._miscentered_deltasigma[i+1,j])
                numpy.testing.assert_approx_equal(nfw_halo._miscentered_deltasigma_table((numpy.log(mean_x), numpy.log(nfw_halo.table_x[j]))), mean_ds)
        
def test_probabilities():
    import scipy.interpolate
    cosmology_obj = fake_cosmo()
    nfw_halo = offset_nfw.NFWModel(cosmology_obj)
    nfw_halo._buildTables()
    nfw_halo._buildRayleighProbabilities(save=False)
    nfw_halo._buildExponentialProbabilities(save=False)

    n = len(nfw_halo.table_x)
    x_min = nfw_halo.x_min
    x_max = nfw_halo.x_max
    random_x = x_min+(x_max-x_min)*numpy.random.random(10)
    random_xmis = x_min+(x_max-x_min)*numpy.random.random(10)
    logr_interval = nfw_halo.table_x[1]/nfw_halo.table_x[0]
    logr_mult = numpy.sqrt(logr_interval)-1./numpy.sqrt(logr_interval)
    dx = logr_mult*nfw_halo.table_x

    this_rayleigh = nfw_halo._rayleigh_p*nfw_halo._rayleigh_orig/dx
    rayleigh_table = scipy.interpolate.RegularGridInterpolator(
        (numpy.log(nfw_halo.table_x), numpy.log(nfw_halo.table_x)),
        this_rayleigh)
    numpy.testing.assert_approx_equal(numpy.sum(nfw_halo._rayleigh_p), n)
    numpy.testing.assert_array_almost_equal(numpy.sum(nfw_halo._rayleigh_p, axis=1), numpy.ones(n))
    numpy.testing.assert_array_almost_equal(rayleigh_table((numpy.log(random_xmis), numpy.log(random_x))),
        random_x/random_xmis**2*numpy.exp(-0.5*(random_x**2/random_xmis**2)))
    # Test integral rayleigh table *x, which should come out to sqrt(pi/2)*x_miscentering.
    # Chop off the bottom quarter and the top quarter, which hit the edges too much
    q = int(0.25*n)
    numpy.testing.assert_array_almost_equal(numpy.sum(nfw_halo.table_x*nfw_halo._rayleigh_p,axis=1)[q:-q],
        numpy.sqrt(numpy.pi/2)*nfw_halo.table_x[q:-q], decimal=4)

    this_exponential = nfw_halo._exponential_p*nfw_halo._exponential_orig/dx
    exponential_table = scipy.interpolate.RegularGridInterpolator(
        (numpy.log(nfw_halo.table_x), numpy.log(nfw_halo.table_x)),
        this_exponential)
    numpy.testing.assert_approx_equal(numpy.sum(nfw_halo._exponential_p), n)
    numpy.testing.assert_array_almost_equal(numpy.sum(nfw_halo._exponential_p, axis=1), numpy.ones(n))
    numpy.testing.assert_array_almost_equal(exponential_table((numpy.log(random_xmis), numpy.log(random_x))),
        random_x/random_xmis**2*numpy.exp(-random_x/random_xmis))
    # Test integral exponential table *x, which should come out to 2*x_miscentering.
    # Chop off the bottom quarter and the top quarter, which hit the edges too much
    numpy.testing.assert_array_almost_equal(numpy.sum(nfw_halo.table_x*nfw_halo._exponential_p,axis=1)[q:-q],
        2*nfw_halo.table_x[q:-q], decimal=4)
        
def test_probability_signal_tables():
    # Spot-check that the *rows* of the miscentered table are the radial dependence of a
    # miscentered profile, properly summed.
    cosmology_obj = fake_cosmo()
    nfw_halo = offset_nfw.NFWModel(cosmology_obj, precision=1)
    nfw_halo._buildTables()
    nfw_halo._buildSigma(save=False)
    nfw_halo._setupSigma()
    nfw_halo._buildMiscenteredSigma(save=False)
    nfw_halo._buildMiscenteredDeltaSigma(save=False)
    nfw_halo._buildRayleighProbabilities(save=False)
    nfw_halo._buildExponentialProbabilities(save=False)
    nfw_halo._buildRayleighSigma(save=False)
    nfw_halo._buildExponentialSigma(save=False)
    nfw_halo._buildRayleighDeltaSigma(save=False)
    nfw_halo._buildExponentialDeltaSigma(save=False)

    n = len(nfw_halo.table_x)
    check_rows = numpy.random.randint(0,n,n/10)
    for row in check_rows:
        test_point = numpy.sum(nfw_halo._exponential_p[row]*nfw_halo._miscentered_sigma[:,1])
        numpy.testing.assert_equal(test_point, nfw_halo._exponential_sigma[row,1]) 
        test_row = numpy.sum(nfw_halo._exponential_p[row, numpy.newaxis]*nfw_halo._miscentered_sigma, axis=1)
        numpy.testing.assert_equal(test_row, nfw_halo._exponential_sigma[row])
        test_point = numpy.sum(nfw_halo._rayleigh_p[row]*nfw_halo._miscentered_sigma[:,1])
        numpy.testing.assert_equal(test_point, nfw_halo._rayleigh_sigma[row,1]) 
        test_row = numpy.sum(nfw_halo._rayleigh_p[row, numpy.newaxis]*nfw_halo._miscentered_sigma, axis=1)
        numpy.testing.assert_equal(test_row, nfw_halo._rayleigh_sigma[row])
        
    nfw_halo._setupRayleighSigma()
    nfw_halo._setupExponentialSigma()
    nfw_halo._setupRayleighDeltaSigma()
    nfw_halo._setupExponentialDeltaSigma()
    for i in range(0,n-1,n/10):
        for j in range(0,n-1,n/10):
            mean_x = numpy.sqrt(nfw_halo.table_x[i]*nfw_halo.table_x[i+1])
            mean_ds = 0.5*(nfw_halo._rayleigh_sigma[i,j]+nfw_halo._rayleigh_sigma[i+1,j])
            numpy.testing.assert_approx_equal(nfw_halo._rayleigh_sigma_table((numpy.log(mean_x), numpy.log(nfw_halo.table_x[j]))), mean_ds)
            mean_ds = 0.5*(nfw_halo._rayleigh_deltasigma[i,j]+nfw_halo._rayleigh_deltasigma[i+1,j])
            numpy.testing.assert_approx_equal(nfw_halo._rayleigh_deltasigma_table((numpy.log(mean_x), numpy.log(nfw_halo.table_x[j]))), mean_ds)
            mean_ds = 0.5*(nfw_halo._exponential_sigma[i,j]+nfw_halo._exponential_sigma[i+1,j])
            numpy.testing.assert_approx_equal(nfw_halo._exponential_sigma_table((numpy.log(mean_x), numpy.log(nfw_halo.table_x[j]))), mean_ds)
            mean_ds = 0.5*(nfw_halo._exponential_deltasigma[i,j]+nfw_halo._exponential_deltasigma[i+1,j])
            numpy.testing.assert_approx_equal(nfw_halo._exponential_deltasigma_table((numpy.log(mean_x), numpy.log(nfw_halo.table_x[j]))), mean_ds)
    
if __name__=='__main__':
    test_object_creation()
    test_scale_radii()
    test_against_colossus()
    test_z_ratios_theory()
    test_against_galsim_theory()
    test_against_clusterlensing_theory()
    test_sigma_to_deltasigma_theory()
    test_g()
    test_Upsilon()
    test_ordering()
    test_setup_table()
    test_build_sigma()
    test_build_miscentered_sigma()
    test_build_deltasigma()
    test_build_miscentered_deltasigma()
    test_probabilities()
    test_probability_signal_tables()

