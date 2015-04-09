#!/usr/bin/env python
from __future__ import division
import numpy as np

def _xdiff(d):
  out = ()
  for itr,val in enumerate(d):
    out += (itr,)*val
  return out

def _cdiff(d,dim):
  out = ()
  for itr in range(dim):
    out += d.count(itr),
  return out
    
def _chain(f,g,x,diff,f_args=None,f_kwargs=None,g_args=None,g_kwargs=None):
  '''
  computes d^n f/(dx_1 dx_2 ... dx_n)
  for f(g(x1,x2,...x_n))
  '''  
  if f_args is None:
    f_args = ()

  if f_kwargs is None:
    f_kwargs = {}

  if g_args is None:
    g_args = ()

  if g_kwargs is None:
    g_kwargs = {}

  #deriv = (int(d) for d in deriv)
  dim = len(diff)
  diff = _xdiff(diff)
  out = np.zeros(np.shape(f(g(x,*g_args,**g_kwargs),*f_args,**f_kwargs)))
  for p in _partitions(diff):
    print(p)
    Np = len(p)
    print([_cdiff(b,dim) for b in p])
    out += (f(g(x,*g_args,**g_kwargs),diff=Np,*f_args,**f_kwargs)
            *np.prod([g(x,*g_args,diff=_cdiff(b,dim),**g_kwargs) for b in p],0))

  return out
      
def _partitions(s):
  '''
  I will generalize this eventually but this is sufficient for now
  '''
  assert len(s) <= 3
  out = ()
  if len(s) == 0:
    out += (),
  if len(s) == 1:
    out += ((s[0],),),
  if len(s) == 2:
    out += ((s[0],),(s[1],)),
    out += ((s[0],s[1]),),
  if len(s) == 3:
    out += ((s[0],),(s[1],),(s[2],)),
    out += ((s[2],),(s[0],s[1])),
    out += ((s[1],),(s[0],s[2])),
    out += ((s[0],),(s[1],s[2])),
    out += ((s[0],s[1],s[2]),),
  return out  

def _r(x,diff=None):
  '''
  function which computes radius from a collection of points or the derivatives
  of radius with respect to various dimensions.
  '''
  if diff is None:
    diff = (0,0,0)

  while len(diff) < 3:
    diff += (0,)    

  if diff == (0,0,0):
    return np.sum(x**2,1)**(1/2)

  elif diff == (1,0,0):
    return x[:,0]/np.sum(x**2,1)**(1/2)

  elif diff == (0,1,0):
    return x[:,1]/np.sum(x**2,1)**(1/2)

  elif diff == (0,0,1):
    return x[:,2]/np.sum(x**2,1)**(1/2)

  elif diff == (2,0,0):
    return (np.sum(x**2,1)-x[:,0]**2)/np.sum(x**2,1)**(3/2)

  elif diff == (0,2,0):
    return (np.sum(x**2,1)-x[:,1]**2)/np.sum(x**2,1)**(3/2)

  elif diff == (0,0,2):
    return (np.sum(x**2,1)-x[:,2]**2)/np.sum(x**2,1)**(3/2)

  elif diff == (1,1,0):
    return -x[:,0]*x[:,1]/np.sum(x**2,1)**(3/2)

  elif diff == (0,1,1):
    return -x[:,1]*x[:,2]/np.sum(x**2,1)**(1/2)

  else:
    print('unsupported derivative for _r: %s' % (diff,))
    return

def _cartesian(fin):
  def fout(x,center,eps=1.0,diff=None):
    '''
    evaluate the radial basis function (RBF) and its derivates

    Parameters
    ----------
      x: (1D or 2D array) locations to evaluate the RBF.  If two dimensional 
        array is given then the first axis length must be the number of points
        and the second axis length must be the number of spatial dimensions

      center: (scalar or 1D) center of the RBF. If 1D array is given then its 
        length must be the number of spatial dimensions which must also equal 
        the length of the second axis for x

      eps: (scalar, default=1.0) scales the width of the RBF.  

      diff: (string, default='') A string indicating the derivatives of the 
        RBF to output.  integer values return derivatives w.r.t the 
        corresponding dimension. 'r' indicates a derivative w.r.t the radius.
        Mixed partials can be computed by concatenating integers.  For example,
        diff='01' will return the mixed partial derivate of the RBF with respect  
        axis zero and axis one.
 
    '''
    # check input
    if len(np.shape(x)) == 1:
      x = x[:,None]
    if len(np.shape(center)) == 0:
      center = np.asarray(center)
      center = center[None]

    assert len(np.shape(x)) == 2
    assert len(np.shape(center)) == 1
    dim = len(center)
    assert dim <= 3
    assert len(diff) <= dim
    assert np.shape(x)[1] == dim
    if diff == None:
      diff = (0,)*dim

    assert sum(diff) <= 3  


    out = _chain(fin,_r,x-center,diff,f_args=(eps,))
    return out

  # set doc string and function name
  fout.__name__ = fin.__name__
  fout.__doc__  = fin.__doc__ + fout.__doc__
  return fout

def _ga(r,eps,diff=0):
  '''
    Gaussian radial basis function ( exp(-r**2) )
  '''
  assert np.all(r == np.abs(r))
  assert diff <= 4 
  r *= eps
  if diff == 4:
    return 4*(4*r**4 - 12*r**2 + 3)*np.exp(-r**2)

  if diff == 3:
    return -4*r*(2*r**2 - 3)*np.exp(-r**2)

  if diff == 2:
    return (4*r**2 - 2)*np.exp(-r**2)

  if diff == 1:
    return -2*r**2*np.exp(-r**2)

  if diff == 0:
    return np.exp(-r**2)

  else:
    print('unsupported derivative for _ga: %s' % diff)

ga = _cartesian(_ga)

def _iq(r,eps,diff=0):
  '''
    Inverse quadratic radial basis function ( 1/(1 + r**2) )
  '''
  assert np.all(r == np.abs(r))
  assert diff <= 4 
  r *= eps
  if diff == 4:
    return 24*(5*r**4 - 10*r**2 + 1)/(r**2 + 1)**5

  elif diff == 3:
    return 24*r*(r**2 - 1)/(r**2 + 1)**4

  elif diff == 2:
    return (6*r**2 - 2)/(1 + r**2)**3

  elif diff == 1:
    return -2*r/(1 + r**2)**2

  elif diff == 0: 
    return 1/(1 + r**2)

  else:
    print('unsupported derivative for _iq: %s' % diff)

iq = _cartesian(_iq)

  
