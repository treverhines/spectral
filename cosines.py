#!/usr/bin/env python
'''
functions used for spectral methods
'''
import numpy as np
import sympy as sp
from scipy.integrate import quadrature

def cosine_term(x,n,a=0.0,b=np.pi,diff=0):
  '''
  returns the function cos(pi*n*(x-a)/(b-a)) or its derivatives 

  PARAMETERS
  ----------
    x: location where the cosine function will be evaluated
    n: order of the cosine function
    a: the left edge of the domain
    b: the right edge of the domain
    diff: (default=0) evaulates the derivative of this order 
  '''
  assert n >= 0
  assert diff >= 0

  x = np.asarray(x)
  scale = np.pi/(b-a)
  x_scale = scale*(x - a)
  if diff%4 == 0:
    return (n*scale)**diff*np.cos(n*x_scale)
  elif diff%4 == 1:
    return ((n*scale)**diff)*(-1)*np.sin(n*x_scale)
  elif diff%4 == 2:
    return ((n*scale)**diff)*(-1)*np.cos(n*x_scale)
  elif diff%4 == 3:
    return (n*scale)**diff*np.sin(n*x_scale)

def cosine_roots(n,a=0.0,b=np.pi):
  '''
  returns the roots of cos(pi*n*(x-a)/(b-a)) on the interval (a,b)
  '''
  roots = np.zeros(n)
  for k in range(n):
    roots[k] = (b-a)*(1.0+2.0*k)/(2.0*n) + a
  return np.asarray(roots)

