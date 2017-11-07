"""utils.py

Includes utility functions for OffsetNFW: at this time, decorators to ensure inputs to the
user-facing functions in OffsetNFW are broadcastable.  These aren't as thoroughly documented as
the user-facing functions in other files, but they should still at least be readable."""

import numpy
import astropy.units as u
from functools import wraps
from inspect import getargspec

def _form_iterables(*args):
    """ Make input arrays broadcastable in the way we want.  We can't just use meshgrid since we may
    have an arbitrary number of vectors of the same length that all need to be tiled."""
    original_args = args
    r = numpy.atleast_1d(args[0])
    args = args[1:]
    # A lot of these "iterability" checks don't work on astropy.Quantities, so un-make the
    # Quantities and then remake them at the end.
    arg_units = [a.unit if hasattr(a, 'unit') else 1 for a in args]
    args = [a.value if hasattr(a, 'value') else a for a in args]
    
    # Don't do anything if everything non-r is a scalar, or if r is a scalar
    is_iterable = [hasattr(a, '__iter__') and len(a)>1 for a in args]
    if sum(is_iterable)==0 or (len(r)==1 and not hasattr(original_args[0], "__iter__")):
        return [numpy.array(o) if hasattr(o,'__iter__') else o for o in original_args]
    # Check everything that isn't r is the same shape
    obj_shapes = []
    for arg, iter in zip(args, is_iterable):
        if iter:
            obj_shapes.append(numpy.array(arg).shape)
    if len(set(obj_shapes))>1:
        raise RuntimeError("All iterable non-r parameters must have same shape")
    # Add an extra axis to the non-r arguments so they're column vectors instead (or the analogue
    # for multi-dimensional arrays, lol).
    args = [numpy.atleast_1d(a) if hasattr(a, '__len__') else a for a in args]
    args = [a[:,numpy.newaxis] if isinstance(a, numpy.ndarray) else a for a in args]
    args = [a*au if a is not None else a for a, au in zip(args, arg_units)]
    return (r,)+tuple(args)

def _form_iterables_multisource(nargs, z_indx, *args):
    """ Make input arrays broadcastable in the way we want, in the case where we have source 
    redshifts to contend with.  Argument nargs is the number of args passed to func; this is 
    assumed to INCLUDE a self argument that won't get passed to this decorator."""
    original_args = args
    z_source = args[z_indx]
    args = args[:z_indx]+args[z_indx+1:]
    r = args[0]
    args = args[1:]
    # A lot of these "iterability" checks don't work on astropy.Quantities, so un-make the
    # Quantities and then remake them at the end.
    arg_units = [a.unit if hasattr(a, 'unit') else 1 for a in args]
    args = [a.value if hasattr(a, 'value') else a for a in args]
    is_iterable = [hasattr(a, '__iter__') and len(a)>1 for a in args]
    r_is_iterable = hasattr(r, '__iter__')
    sz_is_iterable = hasattr(z_source, '__iter__')
#    print args, z_source, original_args, z_indx

    # If only max one of these arguments is iterable, no need to broadcast
    if r_is_iterable+sz_is_iterable+sum(is_iterable)<2:
        if len(original_args)==nargs-1:
            # Ensure consistency with the rest of this fn, which ignores z_source_pdf
            return [numpy.array(o) if hasattr(o,'__iter__') else o for o in original_args[:-1]]
        else:
            return [numpy.array(o) if hasattr(o,'__iter__') else o for o in original_args]
    # or if z_source is not iterable, we can use the function for the 2d broadcasting
    elif sum(is_iterable)==0 or not sz_is_iterable:
        if len(original_args)==nargs-1:
            return _form_iterables(*original_args[:-1])
        else:
            return _form_iterables(*original_args)
    # Check that there's only array shape in the non-r, non-z_source arrays
    obj_shapes = []
    for arg, iter in zip(args, is_iterable):
        if iter:
            obj_shapes.append(numpy.array(arg).shape)
    if len(set(obj_shapes))>1:
        raise RuntimeError("All iterable non-r parameters must have same shape")
    r = numpy.atleast_1d(r)
    if len(r)==1 and not hasattr(original_args[0], '__iter__'): 
        # Okay. So r is a scalar, but not the other arguments. That means we need an extra axis
        # ONLY for the source_z array.
        return (r[0],)+tuple(args)+(z_source[:, numpy.newaxis],)
    # Everything's iterable (or at least one thing in the non-r, non-z_source arrays).
    # Make the args column-like and z_source hypercolumn-like.
    z_source = numpy.atleast_1d(z_source)
    args = [numpy.atleast_1d(a) if hasattr(a, '__len__') else a for a in args]
    args = [a[:,numpy.newaxis] if isinstance(a, numpy.ndarray) else a for a in args]
    args = [a*au if a is not None else a for a, au in zip(args, arg_units)]
    z_source = z_source[:,numpy.newaxis,numpy.newaxis]
    args.insert(z_indx-1, z_source)
    return (r,)+tuple(args)

def reshape(func):
    """This is a decorator to handle reforming input vectors into something broadcastable for easy
    multiplication or interpolation table calls.  Arrays can generally be arbitrary shapes.  
    Pass the kwarg 'skip_reformat' to skip this process (mainly
    used for OffsetNFW object methods that reference other methods)."""
    @wraps(func)
    def wrap_shapes(self, *args, **kwargs):
        skip_reformat = kwargs.pop('skip_reformat', False)
        if skip_reformat:
            return func(self, *args, **kwargs)
        new_args = _form_iterables(*args)
        # all kwargs should have been consumed above
        return func(self, *new_args)
    return wrap_shapes



def reshape_multisource(func):
    """This is a decorator to handle reforming input vectors into something broadcastable for easy
    multiplication or interpolation table calls, in the case where we have a list of source 
    redshifts (args[-2]) and potentially source redshift pdfs (args[-1], which may be None
    to return the full array).  Arrays can generally be arbitrary shapes, but source redshift
    pdfs must be one-dimensional.  Pass the kwarg 'skip_reformat' to skip this process (mainly
    used for OffsetNFW object methods that reference other methods)."""
    @wraps(func)
    def wrap_shapes(self, *args, **kwargs):
        skip_reformat = kwargs.pop('skip_reformat', False)
        if skip_reformat:
            return func(self, *args, **kwargs)
        call_sig = getargspec(func)
        all_args = list(args) +[None]*(len(call_sig.args)-len(args)-1)
        replaced_arg = [True]*len(args)+[False]*(len(call_sig.args)-len(args)-1)
        for key in kwargs:
            ind = call_sig.args.index(key)-1
            if all_args[ind] is not None:
                raise TypeError("Positional and keyword argument passed for %s"%key)
            all_args[ind] = kwargs[key]
            replaced_arg[ind] = True
        for i in range(1,len(call_sig.defaults)+1):
            if all_args[-i] is None and not call_sig.args[-i] in kwargs:
                all_args[-i] = call_sig.defaults[-i]
                replaced_arg[-i] = True
        if numpy.any(replaced_arg==False):
            ind = replaced_arg.index(False)
            raise TypeError("Call to %s missing argument %s"%(func, call_sig.args[ind+1]))
        z_indx = call_sig.args.index('z_source')-1
        zs_indx = call_sig.args.index('z_source_pdf')-1
        zs = all_args[zs_indx]
        if zs is not None and hasattr(zs, '__iter__'):
            zs = numpy.asarray(zs)
        all_args = list(all_args)
        all_args[zs_indx] = None
        
        nargs = len(call_sig.args)
        new_args = _form_iterables_multisource(nargs, z_indx, *all_args)
        # all kwargs should have been consumed above
        result = func(self, *new_args)

        if zs is not None:
            ndim = len(result.shape)
            if hasattr(zs, '__iter__'):
                if hasattr(all_args[z_indx], '__iter__'):
                    if ndim>=3:
                        return numpy.sum(zs[:, numpy.newaxis, numpy.newaxis]*result, axis=0)
                    elif ndim==2:
                        return numpy.sum(zs[:, numpy.newaxis]*result, axis=0)
                    else:
                        return numpy.sum(zs*result)
                else:
                    return numpy.sum(zs)*result
            elif hasattr(all_args[z_indx], '__iter__'):
                return zs*numpy.sum(result, axis=0)
            else:
                return zs*result
        else:
            return result
    return wrap_shapes

