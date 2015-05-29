#!/usr/bin/env python
from __future__ import division
import numpy as np

def bspline_1d(x,k,n,p,diff=0):
  '''
  returns a single 1-D B-spline

  Parameters
  ----------
    x: where the basis function will be evaluated
    k: B-spline knots 
    n: B-spline index 
    p: B-spline order (0 is a step function)

  '''
  assert p >= 0,(
    'received a negative spline order')
  assert n >= 0,(
    'received a negative spline index')
  assert len(k) >= (n+p+2),(
    'there are not enough knots for the given spline order and index')

  tol = 1e-10
  x = np.asarray(x)
  if diff > 0:
    if abs(k[n+p] - k[n]) > tol:
      b1 = p/(k[n+p] - k[n])*bspline_1d(x,k,n,p-1,diff=diff-1)

    else:
      b1 = 0.0*x

    if abs(k[n+p+1] - k[n+1]) > tol:
      b2 = p/(k[n+p+1] - k[n+1])*bspline_1d(x,k,n+1,p-1,diff=diff-1)

    else:
      b2 = 0.0*x

    return b1 - b2

  if p == 0:
    return ((x >= k[n]) & (x < k[n+1])).astype(float)

  else:
    if abs(k[n+p] - k[n]) > tol:
      b1 = (x - k[n])/(k[n+p] - k[n])*bspline_1d(x,k,n,p-1)

    else:
      b1 = 0.0*x

    if abs(k[n+p+1] - k[n+1]) > tol:
      b2 = (x - k[n+p+1])/(k[n+p+1] - k[n+1])*bspline_1d(x,k,n+1,p-1)

    else:
      b2 = 0.0*x

    return b1 - b2

def bspline_nd(x,k,n,p,diff=None):
  '''
  returns an N-D B-spline which is the tensor product of 1-D B-splines

  Parameters
  ----------

    x: points where the b spline will be evaluated this is a (*,N)
      array, where N is the number of spatial dimensions

    k: knots for each dimension. This is a sequence with length N,
      where each element is a list containing the knots for each spatial
      dimension

    n: length N sequence where each element is an integer indicating the 
      B-splines index for each spatial dimension  

    p: order of the B-spline (0 is a step function) 

  '''
  d = len(n)
  if diff is None:
    diff = (0,)*d

  val = [bspline_1d(x[:,i],k[:,i],n[i],p,diff=diff[i]) for i in range(d)]
  return np.prod(val,0)




