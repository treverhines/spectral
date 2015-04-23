#!/usr/bin/env python
import numpy as np
from misc import Timer
from misc import listify
from scipy.interpolate import interp1d
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
from descartes import PolygonPatch

import logging
logger = logging.getLogger(__name__)
  
def nearest(x):
  '''                                                                                                              
  returns a list of distances to the nearest point for each point in
  x. This is used to determine the shape parameter for each radial 
  basis function      
  '''
  tol = 1e-4
  x = np.asarray(x)
  if len(np.shape(x)) == 1:
    x = x[:,None]

  N = len(x)
  A = (x[None] - x[:,None])**2
  A = np.sqrt(np.sum(A,2))
  A[range(N),range(N)] = np.max(A)
  nearest_dist = np.min(A,1)
  nearest_idx = np.argmin(A,1)
  if any(nearest_dist < tol):
    print('WARNING: at least one node is a duplicate or very close to '
          'another node')

  return nearest_dist,nearest_idx


class NodeCollection(dict):
  '''
  keeps track of nodes groups and their indices
  ''' 
  def __init__(self):
    dict.__init__(self)
    self['nodes'] = np.zeros((0,2))       
    self['total'] = 0

  def add(self,nodes,name):    
    nodes = np.asarray(nodes,dtype=float)
    self['nodes'] = np.concatenate((self['nodes'],nodes))

    new_indices = np.arange(self['total'],self['total']+len(nodes))
    self['total'] += len(nodes)

    old_indices = self.get(name,np.zeros(0,dtype=int))
    self[name] = np.concatenate((old_indices,new_indices))


class Domain(Polygon):
  def __init__(self,ext_curves,
                   ext_names,
                   int_curves=None,
                   int_names=None):
    '''
    Define domain by a collection of coordinate pairs

    Parameters
    ----------
      ext_curve: a list of coordinate pairs defining each curve of the domain 
        exterior.  The curves must be ordered such that the end of one curve
        corresponds with the beginning of the next curve

      ext_names: names for each exterior curve

      int_curves: A sequence of objects matching the criterior for ext_curve

      int_names: A sequence of objects matching the criterior for ext_names
    '''  
    if int_curves is None:
      int_curves = []

    if int_names is None:
      int_names = []

    ext_curves = listify(ext_curves)
    ext_names = listify(ext_names)
    int_curves = listify(int_curves)
    int_names = listify(int_names)

    assert len(ext_curves) == len(ext_names)
    assert len(int_curves) == len(int_names)
    assert all(len(c) == len(n) for c,n in zip(int_curves,int_names))

    self.curves = {}
    ext_coords = []
    for c,n in zip(ext_curves,ext_names):
      self.curves[n] = LineString(c)
      ext_coords += c

    int_poly = []
    for C,N in zip(int_curves,int_names):
      int_coords = []
      for c,n in zip(C,N):
        int_coords += c
        self.curves[n] = LineString(c)

      int_poly += [int_coords]
  
    Polygon.__init__(self,ext_coords,int_poly)        
    assert self.is_valid, (
      'domain is not a valid curve. Make sure that no segments '
      'are intersecting')

  def patch(self,*args,**kwargs):
    return PolygonPatch(self,*args,**kwargs)

  def contains(self,points,tol=None):
    '''
    tol gives a buffer area
    '''
    out = np.zeros(len(points))
    if tol is not None:
      is_pos = tol > 0
      line_buffers = []
      for c in self.curves.itervalues():
        line_buffers += [c.buffer(abs(tol))]

      for idx,val in enumerate(points):
        p = Point(val)
        in_buffers = any(l.contains(p) for l in line_buffers)
        in_polygon = Polygon.contains(self,p)
        if is_pos:
          result = in_polygon | in_buffers
        else:
          result = in_polygon & (not in_buffers)

        out[idx] = result

      return out.astype(bool)

    else:
      for idx,val in enumerate(points):
        p = Point(val)
        out[idx] = Polygon.contains(self,p)

      return out.astype(bool)

  def __call__(self,t,name):
    if hasattr(t,'__iter__'):
      out = np.zeros((len(t),2))
      for itr,val in enumerate(t):
        out[itr] = np.array(self.curves[name].interpolate(val,normalized=True))

    else:
      out = np.array(self.curves[name].interpolate(t,normalized=True))

    return out


