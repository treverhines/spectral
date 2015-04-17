#!/usr/bin/env python
import numpy as np
from misc import Timer
from scipy.interpolate import interp1d
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
from descartes import PolygonPatch

import logging
logger = logging.getLogger(__name__)

def listify(a):
  out = []
  for i in a:
    if hasattr(i,'__iter__'):
      out += [listify(i)]
    else:
      out += [i]

  return out
  
class NodeCollection(dict):
  def __init__(self):
    dict.__init__(self)
    self['nodes'] = np.zeros((0,2))       
    self['nearest'] = np.zeros(0)       
    self['total'] = 0

  def add(self,nodes,name):    
    nodes = np.asarray(nodes,dtype=float)
    self['nodes'] = np.concatenate((self['nodes'],nodes))

    new_indices = np.arange(self['total'],self['total']+len(nodes))
    self['total'] += len(nodes)

    old_indices = self.get(name,np.zeros(0,dtype=int))
    self[name] = np.concatenate((old_indices,new_indices))
    #self._compute_nearest()

  def compute_nearest(self):  
    out = np.zeros(self['total'])
    if self['total'] > 1:      
      nidx = range(self['total'])
      for idx in range(self['total']):
        popped = nidx.pop(idx)
        dist = np.sqrt(np.sum((self['nodes'][nidx] - self['nodes'][idx])**2,1))
        min_dist = np.min(dist)
        nidx.insert(popped,idx)
        if min_dist == 0:
          logger.warning(
            'Node %s, %s, is a duplicate node' % (idx,self['nodes'][idx]))
        out[idx] = np.min(dist)

    self['nearest'] = out 
     
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
    assert self.is_valid

  def patch(self,*args,**kwargs):
    return PolygonPatch(self,*args,**kwargs)

  def contains(self,points):
    out = np.zeros(len(points))
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

timer = Timer()
def io_check(fin):
  def fout(self,*args):
    if len(args) == 2:
      is_edge = False
      xy_list = args[0]
      uv_list = args[1]

    elif len(args) == 1:
      edge = args[0]  
      is_edge = isinstance(edge,Edge)
      if is_edge:
        edge = [edge]

      xy_list = np.array([e.xy1 for e in edge]) 
      uv_list = np.array([e.uv for e in edge]) 

    out = fin(self,xy_list,uv_list)
    if is_edge:
      out = out[0]

    return out

  fout.__name__ = fin.__name__
  fout.__doc__ = fin.__doc__
  return fout

class TopologyError(Exception):
  def __init__(self,val):
    self.val = val

  def __str__(self):
    return self.val

class Edge(object):
  def __init__(self,xy1,xy2):
    self.xy1 = np.asarray(xy1)
    self.xy2 = np.asarray(xy2)
    self.uv = self.xy2 - self.xy1

  @io_check
  def is_parallel(self,xy_list,uv_list):
    '''
    returns True when edge is parallel to self
    '''
    a = np.cross(uv_list,self.uv)
    cond1 = (a == 0)          
    return cond1

  @io_check
  def is_collinear(self,xy_list,uv_list):
    '''
    returns True when edge is collinear with self
    '''
    a = np.cross(xy_list-self.xy1,self.uv)             
    cond1 = self.is_parallel(xy_list,uv_list)
    cond2 = (a == 0)
    return cond1 & cond2

  @io_check
  def is_overlapping(self,xy_list,uv_list):
    '''
    returns True if there is a finite width of overlap between edge and self.
    Overlapping vertices do not count as an overlap
    '''
    a = np.sum((xy_list-self.xy1)*self.uv,1)
    b = np.sum((self.xy1-xy_list)*uv_list,1)
    c = np.dot(self.uv,self.uv)
    d = np.sum(uv_list*uv_list,1)
    cond1 = self.is_collinear(xy_list,uv_list)
    cond2 = (a > 0) & (a < c)
    cond3 = (b > 0) & (b < d)
    return cond1 & (cond2 | cond3)
      

  @io_check
  def is_intersecting(self,xy_list,uv_list):
    '''
    returns True if there is any point intersection between edge and self.
    Overlapping vertices does not count as an intersection.
    '''
    cond1 = self.is_parallel(xy_list,uv_list) == False
    xy_list = xy_list[cond1]
    uv_list = uv_list[cond1]

    a = np.cross(xy_list-self.xy1,uv_list)
    b = np.cross(self.xy1-xy_list,self.uv)
    c = np.cross(self.uv,uv_list)
    t = a/c
    u = -b/c
    cond2 = (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    cond1[cond1] = cond2
    return cond1

  @io_check
  def intersection(self,xy_list,uv_list):
    '''
    returns the point of intersection between edge and self if it exists
    '''
    if any(self.is_intersecting(xy_list,uv_list) == False):
      raise TopologyError('an intersection point does not exist')

    a = np.cross(xy_list-self.xy1,uv_list)
    b = np.cross(self.xy1-xy_list,self.uv)    
    c = np.cross(self.uv,uv_list)
    d = np.cross(uv_list,self.uv) 
    t = a/c
    u = b/d
    return self.xy1 + t[:,None]*self.uv

class JordanCurve(object):
  def __init__(self,points):
    points = np.asarray(points)
    assert len(np.shape(points)) == 2
    assert np.shape(points)[1] == 2
    N = np.shape(points)[0]
    self.edges = []
    for i in range(1,N):
      e = Edge(points[i-1,:],points[i,:])
      if i > 1:
        if any(e.is_overlapping(self.edges)):
          raise TopologyError('creating an edge with points %s and %s causes '
                              'an overlap' % (points[i-1],points[i]))

      if (i > 2):
        if any(e.is_intersecting(self.edges[:-1])):
          raise TopologyError('creating an edge with points %s and %s causes '
                              'an intersection' % (points[i-1],points[i]))

      self.edges += [e]

    e = Edge(points[-1,:],points[0,:])
    if any(e.is_overlapping(self.edges)):
      raise TopologyError('creating an edge with points %s and %s causes an '
                          'overlap' % (points[-1],points[0]))

    if any(e.is_intersecting(self.edges[1:-1])):
      raise TopologyError('creating an edge with points %s and %s causes an '
                          'intersection' % (points[-1],points[0]))

    self.edges += [e]

    self.x = np.array([e.xy1[0] for e in self.edges]+[self.edges[0].xy1[0]])
    self.y = np.array([e.xy1[1] for e in self.edges]+[self.edges[0].xy1[1]])
    edge_length = np.sqrt(np.diff(self.x)**2 + np.diff(self.y)**2)
    total_length = np.array([np.sum(edge_length[:i]) for i in range(N+1)])
    t = total_length/total_length[-1]               
    self.xinterp = interp1d(t,self.x)
    self.yinterp = interp1d(t,self.y)

  def contains(self,points,outside=None):
    if outside is None:
      outside = [1e10,1e10]

    if len(np.shape(points)) == 1:
      line = Edge(outside,point)
      count = np.sum(line.is_intersecting(self.edges))
      return  bool(count%2)

    else:
      out = np.zeros(len(points))
      for idx,p in enumerate(points):
        line = Edge(outside,p)
        count = np.sum(line.is_intersecting(self.edges))
        out[idx] = count%2

      return out.astype(bool)

  def __call__(self,t):
    return np.array([self.xinterp(t),self.yinterp(t)]).transpose()


    

