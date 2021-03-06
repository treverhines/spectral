#!/usr/bin/env python
'''
functions used for spectral methods
'''
from __future__ import division
import numpy as np
import sympy as sp

def chebyshev(x,n,a=-1,b=1,diff=0):
  '''
  returns the Chebyshev polynomial of the first kind of degree n
  evaluated at points x.  The polynomial is transformed such that it
  is bounded by +/- 1 on the interval (a,b) rather than (-1,1).  The
  derivatives are approriately scaled according to the chain rule
                                                                             
  The n'th order Chebyshev polynomial is defined as:                         
                                                                             
    T_n(cos(t)) := cos(n*t)                                          
                                                                             
  PARAMETERS:
    x: a vector of points where the polynomial will be evaluated
    a: left edge of the polynomial domain
    b: right edge of the polynomial domain
    n: order of the polynomial

    diff: (default = 0) returns the specified derivative of the
          Chebyshev polynomial.  This must be <= 4

  '''
  x = np.asarray(x)
  assert n >= 0
  assert b > a
  assert diff <= 4
  assert all((x >= a) & (x <= b))

  # handle change of coordinates
  scale = 2/(b-a)
  shift = -(a+b)/(b-a)
  x = scale*x + shift

  tol = 1e-8 # how close x needs to be to a or b before the expression
             # for the nth derivative at the endpoints is invoked

  out = np.zeros(len(x),dtype=np.float64)

  # handle the potentially numerically unstable values at the ends of
  # the domain
  is_a = np.abs(x + 1) < tol
  is_b = np.abs(x - 1) < tol
  is_internal = (x > -1) & (x < 1) & (is_a == False) & (is_b == False)

  out[is_a] = chebyshev_boundary_values(n,diff)[0]
  out[is_b] = chebyshev_boundary_values(n,diff)[1]

  # compute the internal values
  xi = x[is_internal]
  t = np.arccos(xi)
  if diff == 0:
    out[is_internal] = np.cos(t*n)
    return out

  elif diff == 1:
    out[is_internal] = (
      scale*n*np.sin(n*t)/np.sqrt(-xi**2 + 1))
    return out

  elif diff == 2:
    out[is_internal] = (
      scale**2*(n**2*np.cos(n*t)/(xi**2 - 1) + 
      n*xi*np.sin(n*t)/(-xi**2*np.sqrt(-xi**2 + 1) + 
      np.sqrt(-xi**2 + 1))))
    return out

  elif diff == 3:
    out[is_internal] = (
      scale**3*(-n**3*np.sin(n*t)/(-xi**2*np.sqrt(-xi**2 + 1) + 
      np.sqrt(-xi**2 + 1)) - 
      3*n**2*xi*np.cos(n*t)/(xi**4 - 2*xi**2 + 1) + 
      3*n*xi**2*np.sin(n*t)/(xi**4*np.sqrt(-xi**2 + 1) - 
      2*xi**2*np.sqrt(-xi**2 + 1) + np.sqrt(-xi**2 + 1)) +
      n*np.sin(n*t)/(-xi**2*np.sqrt(-xi**2 + 1) + 
      np.sqrt(-xi**2 + 1))))
    return out

  elif diff == 4:
    out[is_internal] = (
      scale**4*(n**4*np.cos(n*t)/(xi**4 - 2*xi**2 + 1) - 
      6*n**3*xi*np.sin(n*t)/(xi**4*np.sqrt(-xi**2 + 1) - 
      2*xi**2*np.sqrt(-xi**2 + 1) + 
      np.sqrt(-xi**2 + 1)) + 
      15*n**2*xi**2*np.cos(n*t)/(xi**6 - 3*xi**4 + 3*xi**2 - 1) -
      4*n**2*np.cos(n*t)/(xi**4 - 2*xi**2 + 1) + 
      15*n*xi**3*np.sin(n*t)/(-xi**6*np.sqrt(-xi**2 + 1) + 
      3*xi**4*np.sqrt(-xi**2 + 1) - 
      3*xi**2*np.sqrt(-xi**2 + 1) + 
      np.sqrt(-xi**2 + 1)) + 
      9*n*xi*np.sin(n*t)/(xi**4*np.sqrt(-xi**2 + 1) - 
      2*xi**2*np.sqrt(-xi**2 + 1) + 
      np.sqrt(-xi**2 + 1))))
    return out
    
def chebyshev_boundary_values(n,diff):
  '''
  evaluated the value of the nth order chebyshev polynomial at -1
  and 1

  '''  
  upper = np.prod([(n**2-k**2)/(2*k+1.0) for k in range(diff)])
  lower = (-1)**(diff+n)*upper
  return (lower,upper)

def chebyshev_poly(x,n,a=-1,b=1,diff=0):
  '''returns the Chebyshev polynomial of the first kind of degree n
  evaluated at points x using the recursive formulation.  This
  function is far slower than the above function and is likely less
  numerically stable.  One potentially nice aspect of this function is
  that a symbolic input for x would return a symbolic polynomial term.
                                                                                   
  The n'th order Chebyshev polynomial is defined as:                               
                                                                                   
    T_n(cos(t)) := cos(n*t)                                                        
                                                                                   
  using the identity                                                               
                                                                                   
    2cos(a)cos(b) = cos(a-b) + cos(a + b)                                          
                                                                                   
  and letting                                                                      
                                                                                   
    a = (n-1)*t; b = t                                                             
                                                                                   
  we get:                                                                          
                                                                                   
    cos(n*t) = 2*cos(t)*cos((n-1)*t) - cos((n-2)t)
  
  then substituting x = cos(theta) and the definition of the Chebyshev
  polynomial into the above equation gives us the following recursive
  formula for the n'th order Chebyshev polynomial, which is what this
  function evaluates
                                                                                   
    T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)                                           
                                                         
  The above expression is what this function evaluates
  
  PARAMETERS:
    x: can be either a scalar, vector, or symbolic variable which
       represents the domain of the chebyshev polynomial.  If not
       symbolic, this needs to be between -1 and 1

    n: order of the polynomial

    diff: (optional) returns the specified derivative of the Chebyshev
          polynomial.  This is computed with a finite difference
          method and is unstable for values greater than 3

  '''
  x = np.asarray(x)
  x = (x-a)*2/(b - a) - 1 

  dx = 1e-3  # the spacing used for the finite difference
             # approximation of the nth derivative. This needs to be
             # less than or equal to tol.
  assert n >= 0
  assert b > a

  if diff == 0:
    if n == 0:
      return 0*x + 1
    if n == 1:
      return x
    else:
      return 2*x*chebyshev_poly(x,n-1) - chebyshev_poly(x,n-2)
  else:
    u_1 = chebyshev_poly(x+dx,n,a,b,diff-1)
    u_0 = chebyshev_poly(x,n,a,b,diff-1)
    return (u_1 - u_0)/dx

def chebyshev_roots(n,a=-1,b=1):
  '''
  returns the roots of the n'th chebyshev polynomial. Specifically,
  this solves for the values of x which satisfy
                                                                                
    T_n(x) = cos(n*acos(x)) = 0                                                 
                                                                                
  which are                                                                     
    x = -cos(((2i-1)pi)/(2*(N))) for i in (1:N)

  '''
  assert n >= 0
  assert b > a
  roots = np.zeros(n)
  for i in range(1,n+1):
    roots[i-1]  = (-np.cos(((2*i-1)*np.pi)/(2*n)) + 1)*(b-a)/2 + a
  return np.asarray(roots)
