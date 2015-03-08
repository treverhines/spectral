#!/usr/bin/env python
'''
functions used for spectral methods
'''
import numpy as np

def exponential_term(x,n,a=0.0,b=np.pi,diff=0):
  '''
  returns the function exp(i*pi*n*(x-a)/(b-a)) or its derivatives 

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
  i = complex(0,1.0)
  exponent = i*n*np.pi*(x-a)/(b-a)
  return (i*n*np.pi/(b-a))**diff*np.exp(exponent) 

def cosine_roots(n,a=0.0,b=np.pi):
  '''
  returns the roots of cos(pi*n*(x-a)/(b-a)) on the interval (a,b)
  '''
  roots = np.zeros(n)
  for k in range(n):
    roots[k] = (b-a)*(1.0+2.0*k)/(2.0*n) + a
  return np.asarray(roots)

