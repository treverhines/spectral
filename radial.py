#!/usr/bin/env python
from __future__ import division
import sympy 
import numpy as np

_R = sympy.symbols('R')
_EPS = sympy.symbols('EPS')

class RBF(object):
  def __init__(self,expr):    
    assert _R in expr, (
      'RBF expression does not contain _R')
    
    assert _EPS in expr, (
      'RBF expression does not contain _EPS')
    self.R_expr = expr
    self.diff_dict = {}
   
  def __call__(self,x,c,eps=1.0,diff=None):
    x = np.asarray(x)
    c = np.asarray(c)    
    if len(np.shape(x)) == 1:
      x = x[:,None]

    if len(np.shape(c)) == 0:
      c = c[None]

    assert len(np.shape(x)) == 2, (
      'x must be either a 1 or 2 dimensional array')   
    assert len(np.shape(c)) == 1, (
      'center must be a scalar or 1 dimensional array')
    dim = np.shape(x)[1]
    assert np.shape(c)[0] == dim, (
      'dimensions of center must equal the dimensions of x')

    if diff is None:
      diff = (0,)*dim

    while len(diff) < dim:
      diff += (0,)

    assert len(diff) == dim, (
      'cannot specify derivatives for dimensions that are higher than the '
      'dimensions of x and center')

    if diff not in self.diff_dict:
      self._make_function(diff)

    return self.diff_dict[diff](*(tuple(x.transpose())+tuple(c)+(eps,)))

  def _make_function(self,diff):
    dim = len(diff)
    c_sym = sympy.symbols('c:%s' % dim)
    x_sym = sympy.symbols('x:%s' % dim)    
    r = sympy.sqrt(sum((x_sym[i]-c_sym[i])**2 for i in range(dim)))
    expr = self.R_expr.subs(_R,r)            
    for direction,order in enumerate(diff):
      if order == 0:
        continue
      expr = expr.diff(*(x_sym[direction],)*order)

    self.diff_dict[diff] = sympy.lambdify(x_sym+c_sym+(_EPS,),expr,'numpy')

_FUNCTION_DOC = '''
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
                                                                           
    diff: (tuple, default=(0,)*dim) a tuple whos length is equal to the number 
      of spatial dimensions.  Each value in the tuple must be an integer
      indicating the order of the derivative in that spatial dimension.  For 
      example, if the the spatial dimensions of the problem are 3 then 
      diff=(2,0,1) would compute the second derivative in the first dimension
      and the first derivative in the third dimension.

  Returns
  -------
    out: 1D array with length equal to the first dimension of x

  Note
  ----
    the derivatives are computed symbolically in Sympy and then lambdified to 
    evaluate the expression with the provided values.  The lambdified functions
    are stored in the scope of the radial module and will be recalled if 
    a value for diff is used more than once in the Python session.        
'''

_IQ = RBF(1/(1+(_EPS*_R)**2))
def iq(*args,**kwargs):
  '''                                                                                                            
  Inverse Quadratic
  '''                                                             
  return _IQ(*args,**kwargs)

iq.__doc__ += _FUNCTION_DOC

_GA = RBF(sympy.exp(-(_EPS*_R)**2))
def ga(*args,**kwargs):
  '''                                                                                                            
  Gaussian
  '''
  return _GA(*args,**kwargs)

ga.__doc__ += _FUNCTION_DOC
