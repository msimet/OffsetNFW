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
    shape = utils._fix_shapes(w,x,y,z)
    numpy.testing.assert_equal(shape, (2,3))
    shape = utils._fix_shapes(w,x,z)
    numpy.testing.assert_equal(shape, (3,))
    shape = utils._fix_shapes(w,x,y,y,z)
    numpy.testing.assert_equal(shape, (2,3))
    numpy.testing.assert_raises(TypeError, utils._fix_shapes, w, x, w, y, z)
    numpy.testing.assert_raises(TypeError, utils._fix_shapes, [w,w], x, y, z)
    numpy.testing.assert_raises(TypeError, utils._fix_shapes, w, x, [y,y], z)

    w_arr = numpy.array([w,w])
    y_arr = numpy.array([y,y,y]).T
    x_arr = numpy.full(w_arr.shape, x)
    z_arr = numpy.full(w_arr.shape, z)
    
    numpy.testing.assert_equal(utils._form_iterables(w,x,y,z), [w_arr, x_arr, y_arr, z_arr])
    numpy.testing.assert_equal(utils._form_iterables(w,x,y,y), [w_arr, x_arr, y_arr, y_arr])
    w2_arr = numpy.array([w,w,w])
    x2_arr = numpy.full(w2_arr.shape, x)

    def func(w,x,y,z):
        return w*x*y*z
    output_arr = func(w_arr,x_arr,y_arr,z_arr)
    numpy.testing.assert_equal(utils._reformat_shape(output_arr, w_arr.shape).shape, (2,3))
    output_arr = func(w,x,x,x)
    numpy.testing.assert_equal(utils._reformat_shape(output_arr, w.shape).shape, (3,))
    
def test_multi_pieces():
    w = numpy.array([1,2,3])
    x = 2.
    y = numpy.array([3,4])
    z = 2.5
    zs = numpy.array([5,6,7,8,9])
    shape = utils._fix_shapes_multisource(7,w,x,y,z,zs,None)
    numpy.testing.assert_equal(shape, (2,3,5))
    shape = utils._fix_shapes_multisource(7,w,x,y,z,zs,zs)
    numpy.testing.assert_equal(shape, (2,3,5))
    shape = utils._fix_shapes_multisource(5,w,x,zs,None)
    numpy.testing.assert_equal(shape, (3,5))
    shape = utils._fix_shapes_multisource(7,w,x,y,y,zs,None)
    numpy.testing.assert_equal(shape, (2,3,5))
    numpy.testing.assert_raises(TypeError, utils._fix_shapes_multisource, 7, w, x, w, y, zs, None)
    numpy.testing.assert_raises(TypeError, utils._fix_shapes_multisource, 6, [w,w], x, y, zs, None)
    numpy.testing.assert_raises(TypeError, utils._fix_shapes_multisource, 6, w, x, [y,y], zs, None)
    numpy.testing.assert_raises(RuntimeError, utils._fix_shapes_multisource, 6, w, x, y, zs, w)

    w_arr, y_arr, zs_arr = numpy.meshgrid(w, y, zs)
    x_arr = numpy.full(w_arr.shape, x)
    z_arr = numpy.full(w_arr.shape, z)    
    
    numpy.testing.assert_equal(utils._form_iterables_multisource(7,w,x,y,z,zs,None), [w_arr, x_arr, y_arr, z_arr, zs_arr])
    numpy.testing.assert_equal(utils._form_iterables_multisource(7,w,x,y,z,zs,zs), [w_arr, x_arr, y_arr, z_arr, zs_arr])
    numpy.testing.assert_equal(utils._form_iterables_multisource(7,w,x,y,y,zs,None), [w_arr, x_arr, y_arr, y_arr, zs_arr])
    w2_arr = numpy.array([w,w,w])
    x2_arr = numpy.full(w2_arr.shape, x)

    def func(w,x,y,z):
        return w*x*y*z
    output_arr = func(w_arr,x_arr,y_arr,zs_arr)
    numpy.testing.assert_equal(utils._reformat_shape(output_arr, w_arr.shape).shape, (2,3,5))
    output_arr = func(w.reshape(3,1),x,x,zs.reshape(1,5))
    numpy.testing.assert_equal(utils._reformat_shape(output_arr, (3,5)).shape, (3,5))
    
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
    ans = numpy.array([[[6., 12., 18., 24., 30.],
                        [12., 24., 36., 48., 60.],
                        [18., 36., 54., 72., 90.]],
                       [[8., 16., 24., 32., 40.],
                        [16., 32., 48., 64., 80.],
                        [24., 48., 72., 96., 120.]]])
    numpy.testing.assert_equal(func(None,w,x,y,z,None), ans)
    numpy.testing.assert_equal(func(None,w,x,y,z,zs), numpy.sum(zs*ans, axis=2))

if __name__=='__main__':
    test_single_pieces()
    test_multi_pieces()
    test_reshape()
    test_reshape_multisource()

