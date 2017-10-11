import sys
sys.path.append('..')
import numpy
import numpy.testing
from offset_nfw import utils

def test_single_pieces():
    w = numpy.array([1,2,3])
    x = 2.
    y = numpy.array([3,4])
    z = 2.5

    w_arr = w
    y_arr = numpy.array([[3],[4]])
    x_arr = x
    z_arr = z
    
    numpy.testing.assert_equal(utils._form_iterables(w,x,y,z), [w_arr, x_arr, y_arr, z_arr])
    numpy.testing.assert_equal(utils._form_iterables(w,x,y,y), [w_arr, x_arr, y_arr, y_arr])
    numpy.testing.assert_equal(utils._form_iterables(w[0],x,y,z), [w[0], x, y, z])
    numpy.testing.assert_equal(utils._form_iterables(w[0:1],x,y,z), [w[0:1], x_arr, y_arr, z_arr])

    
def test_multi_pieces():
    w = numpy.array([1,2,3])
    x = 2.
    y = numpy.array([3,4])
    z = 2.5
    zs = numpy.array([5,6,7,8,9])

    w_arr = w
    x_arr = x
    z_arr = z
    y_arr = numpy.array([[3],[4]])
    zs_arr = numpy.array([[[5]],[[6]],[[7]],[[8]],[[9]]])

    numpy.testing.assert_equal(utils._form_iterables_multisource(7,w,x,y,z,zs,None), [w_arr, x_arr, y_arr, z_arr, zs_arr])
    numpy.testing.assert_equal(utils._form_iterables_multisource(7,w[0],x,y,z,zs,None), [w_arr[0], x_arr, y, z_arr, zs_arr.reshape((-1,1))])
    numpy.testing.assert_equal(utils._form_iterables_multisource(7,w[0:1],x,y,z,zs,None), [w_arr[0:1], x_arr, y_arr, z_arr, zs_arr])
    numpy.testing.assert_equal(utils._form_iterables_multisource(7,w[0],x,y,z,zs[0],None), [w_arr[0], x, y, z, zs[0]])
    numpy.testing.assert_equal(utils._form_iterables_multisource(7,w[0],x,y[0],z,zs,None), [w_arr[0], x, y[0], z, zs])
    numpy.testing.assert_equal(utils._form_iterables_multisource(7,w,x,y,z,zs,zs), [w_arr, x_arr, y_arr, z_arr, zs_arr])
    numpy.testing.assert_equal(utils._form_iterables_multisource(7,w,x,y,y,zs,None), [w_arr, x_arr, y_arr, y_arr, zs_arr])
    
def test_reshape():
    @utils.reshape
    def func(self,w,x,y,z):
        return w*x*y*z
    
    w = numpy.array([1,2,3])
    x = 2
    y = numpy.array([3,4])
    z = 2.5
    
    # We want this to return...
    ans = numpy.array([[15.,30.,45.],
                       [20.,40.,60.]])
    numpy.testing.assert_equal(func(None,w,x,y,z), ans)
    
def test_reshape_multisource():
    @utils.reshape_multisource
    def func(self,w,x,y,z,z_source_pdf=None):
        return w*x*y*z
    
    w = numpy.array([1,2,3])
    x = 2
    y = numpy.array([3,4])
    z = numpy.array([1,2,3,4,5])
    zs = numpy.array([0.25, 0.5, 0.75, 0.25, 0.9])
    
    # We want this to return...
    ans = numpy.array([[[6,12,18],
                        [8,16,24]],
                       [[12,24,36],
                        [16,32,48]],
                       [[18,36,54],
                        [24,48,72]],
                       [[24,48,72],
                        [32,64,96]],
                       [[30,60,90],
                        [40,80,120]]])
    numpy.testing.assert_equal(func(None,w,x,y,z,None), ans)
    numpy.testing.assert_equal(func(None,w,x,y,z,zs), numpy.sum(zs.reshape((-1,1,1))*ans, axis=2))
    numpy.testing.assert_equal(func(None,w,x,y,z,z_source_pdf=zs), 
                               numpy.sum(zs.reshape((-1,1,1))*ans, axis=2))

if __name__=='__main__':
    test_single_pieces()
    test_multi_pieces()
    test_reshape()
    test_reshape_multisource()

