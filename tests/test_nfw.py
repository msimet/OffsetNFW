import numpy
import numpy.testing

def test_form_iterables():
    """ Test the piece that makes iterables out of input parameters """
    pass

def test_scale_radii():
    """ Test scale radius measurement. """
    pass

def test_massdefinitions():
    """ Test different overdensities """
    pass    
    
def test_against_galsim():
    """ Test against the GalSim implementation of NFWs. """
    try:
        import galsim
    except ImportError:
        import warnings
        warnings.warn("Could not test against GalSim -- import failure")
        return True
    
def test_against_cluster_lensing():
    """ Test against the cluster-lensing implementation of NFWs. """
    try:
        import clusterlensing
    except ImportError:
        import warnings
        warnings.warn("Could not test against cluster-lensing -- import failure")

def test_sigma_to_deltasigma():
    """ Test that the numerical sigma -> deltasigma produces the theoretical DS. """
    pass

def test_shear_ratios():
    """ Test that the theoretical shear changes properly with redshift """
    pass
    
def test_convergence_ratios():
    """ Test that the theoretical convergence changes properly with redshift """
    pass
    
def test_g():
    """ Test that the theoretical returned g is the appropriate reduced shear """
    pass
    
def test_Upsilon():
    """ Test that the theoretical Upsilon is the appropriate value. """
    pass
    
def test_table():
    """ Generate a small interpolation table and test that it produces the right results. """
    pass


