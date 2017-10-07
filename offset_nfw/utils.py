import numpy
import astropy.units as u
from inspect import getargspec

def _fix_shapes(*args):
    """
    Figure out the return shape for a set of args, where the first is assumed to be radial and the
    rest are assumed to be either scalars or vectors of len nobj."""
    r = args[0]
    args = args[1:]
    if hasattr(r, '__iter__'):
        r = numpy.atleast_1d(r)
        if len(r.shape)>1:
            raise TypeError("r must be a 1-d vector, not a multi-dimensional array")
        shape = list(r.shape)
    else:
        shape = []
    args_1d = [numpy.atleast_1d(a) for a in args]
    lengths = [(a.shape) for a in args_1d]
    maxlen = max(lengths)
    if any([len(l)>1 for l in lengths]):
        raise TypeError("Parameters passed to OffsetNFW models must be scalars or 1-d vectors, "+
                        "not multi-dimensional arrays")
    set_lengths = set(lengths)
    set_lengths.discard((1,))
    if len(set_lengths)>1:
        raise TypeError("Length of non-r parameter vectors must match")
    if numpy.product(maxlen)==1:
        is_iterable = [hasattr(a, '__iter__') and len(a)>1 for a in args]
        if numpy.any(is_iterable):
            shape += list(maxlen)
    else:
        shape += list(maxlen)
    return tuple(reversed(shape))

def _form_iterables(*args):
    """ Tile the given inputs for different outputs such that we can make a single call to
    the interpolation table.  We can't just use meshgrid since we may have an arbitrary number
    of vectors of the same length that all need to be tiled."""
    # TODO: check this works for multidimensional *args.
    r = args[0]
    args = args[1:]
    is_iterable = [hasattr(a, '__iter__') and len(a)>1 for a in args]
    if sum(is_iterable)==0:
        new_tuple = (r,)+args
        return new_tuple
    obj_shapes = []
    for arg, iter in zip(args, is_iterable):
        if iter:
            obj_shapes.append(numpy.array(arg).shape)
    if len(set(obj_shapes))>1:
        raise RuntimeError("All iterable non-r parameters must have same shape")
    r = numpy.atleast_1d(r)
    args = [a if not hasattr(a, '__iter__') else (a if len(a)>1 else a[0]) for a in args]
    iter_indx = numpy.where(is_iterable)[0][0]
    arg = args[iter_indx]
    shape = (-1,) + r.shape
    temp_arg = numpy.tile(arg, r.shape).reshape(shape)
    new_r = numpy.tile(r, arg.shape).T
    shape = temp_arg.shape
    new_args = [new_r]
    for arg, iter in zip(args, is_iterable):
        if iter:
            new_args.append(numpy.tile(arg, r.shape).reshape(shape[::-1]).T)
        else:
            new_args.append(numpy.tile(arg, shape))
    new_args = [n.reshape(shape) for n in new_args]
    return new_args

def _reformat_shape(array, shape):
    """Take the output of something run on args run through _form_iterables, and reformat to a
    sensible output dimension."""
    if not shape:
        while hasattr(array, 'shape') and array.shape:
            array = array[0]
    else:
        array = array.reshape(shape)
    if isinstance(array, u.Quantity):
        array = array.decompose()
        if array.unit == u.dimensionless_unscaled:
            return array.value
    return array

def _fix_shapes_multisource(nargs, *args):
    """
    Figure out the return shape for a set of args, where the first is assumed to be radial and the
    rest are assumed to be either scalars or vectors of len nobj, up to the last two args, which 
    may have a different shape (but must be the same shape).  The final argument may be None. This
    is to satisfy cases where we either want the same arrays as in _fix_shapes, but for a set of
    different source redshifts (last argument is None), or where we want to integrate over a 
    source redshift distribution (last argument has the same shape as second-to-last argument)."""
    r = args[0]
    if len(args)==nargs-1:
        z_source = args[-2]
        z_source_pdf = args[-1]
        args = args[1:-2]
    else:
        z_source = args[-1]
        z_source_pdf = None
        args = args[1:-1]
    if z_source_pdf is not None and hasattr(z_source_pdf, '__iter__'):
        if not numpy.atleast_1d(z_source_pdf).shape == numpy.atleast_1d(z_source).shape:
            raise RuntimeError('z_source and z_source_pdf must have same shape')
    if hasattr(r, '__iter__'):
        r = numpy.atleast_1d(r)
        if len(r.shape)>1:
            raise TypeError("r must be a 1-d vector, not a multi-dimensional array")
        shape = list(r.shape)
    else:
        shape = []
    args_1d = [numpy.atleast_1d(a) for a in args]
    lengths = [(a.shape) for a in args_1d]
    maxlen = max(lengths)
    if any([len(l)>1 for l in lengths]):
        raise TypeError("Parameters passed to OffsetNFW models must be scalars or 1-d vectors, "+
                        "not multi-dimensional arrays")
    set_lengths = set(lengths)
    set_lengths.discard((1,))
    if len(set_lengths)>1:
        raise TypeError("Length of non-r parameter vectors must match")
    if numpy.product(maxlen)==1:
        is_iterable = [hasattr(a, '__iter__') and len(a)>1 for a in args]
        if numpy.any(is_iterable):
            shape += list(maxlen)
    else:
        shape += list(maxlen)
    shape.reverse()
    if hasattr(z_source, '__iter__'):
        z_source = numpy.atleast_1d(z_source)
        if len(z_source.shape)>1:
            raise TypeError("source redshifts must be a 1-d vector, not a multi-dimensional array")
        shape += list(z_source.shape)
    return tuple(shape)

def _form_iterables_multisource(nargs, *args):
    """ Tile the given inputs for different outputs such that we can make a single call to
    the interpolation table, in the case where we have source redshifts to contend with."""
    original_args = args
    r = args[0]
    if len(args)==nargs-1:
        z_source = args[-2]
        args = args[1:-2]
    else:
        z_source = args[-1]
        args = args[1:-1]
    is_iterable = [hasattr(a, '__iter__') and len(a)>1 for a in args]
    r_is_iterable = hasattr(r, '__iter__')
    sz_is_iterable = hasattr(z_source, '__iter__')

    if r_is_iterable+sz_is_iterable+sum(is_iterable)<2:
        return original_args
    elif sum(is_iterable)==0 or not sz_is_iterable:
        return _form_iterables(original_args)
    
    obj_shapes = []
    for arg, iter in zip(args, is_iterable):
        if iter:
            obj_shapes.append(numpy.array(arg).shape)
    if len(set(obj_shapes))>1:
        raise RuntimeError("All iterable non-r parameters must have same shape")
    r = numpy.atleast_1d(r)
    z_source = numpy.atleast_1d(z_source)
    args = [a if not hasattr(a, '__iter__') else (a if len(a)>1 else a[0]) for a in args]
    iter_indx = numpy.where(is_iterable)[0][0]
    arg = args[iter_indx]
    new_r, new_arg, new_z_source = numpy.meshgrid(r, arg, z_source)
    new_args = [new_r]
    for arg, iter in zip(args, is_iterable):
        if iter:
            new_args.append(numpy.tile(arg, new_arg.shape[1:]).reshape(new_arg.T.shape).T)
        else:
            new_args.append(numpy.tile(arg, new_arg.shape))
    new_args += [new_z_source]
    return new_args

def reshape(func):
    """This is a decorator to handle reforming input vectors into 2-D arrays for easy
    multiplication or interpolation table calls."""
    def wrap_shapes(self, *args, **kwargs):
        shape = _fix_shapes(*args)
        new_args = _form_iterables(*args)
        result = func(self, *new_args, **kwargs)
        return _reformat_shape(result, shape)
    return wrap_shapes

def reshape_multisource(func):
    """This is a decorator to handle reforming input vectors into 2-D arrays for easy
    multiplication or interpolation table calls, in the case where we have a list of source 
    redshifts (args[-2]) and potentially source redshift pdfs (args[-1], which may be None
    to return the full array)."""
    def wrap_shapes(self, *args, **kwargs):
        nargs = len(getargspec(func).args)
        shape = _fix_shapes_multisource(nargs, *args)
        new_args = _form_iterables_multisource(nargs, *args)
        result = _reformat_shape(func(self, *new_args, **kwargs), shape)

        if len(args)==nargs-1 and args[-1] is not None:
            zs = args[-1]
            return numpy.sum(zs*result, axis=2)
        else:
            return result
    return wrap_shapes

