#!/usr/bin/env python
import numpy as np
from misc import Timer
from scipy.interpolate import interp1d
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
    t = np.linspace(0,1,N+1)
    self.xinterp = interp1d(t,self.x)
    self.yinterp = interp1d(t,self.y)

  def contains(self,points):
    minx = min(self.x)
    if len(np.shape(points)) == 1:
      line = Edge([minx,point[1]],point)
      count = np.sum(line.is_intersecting(self.edges))
      return  bool(count%2)

    out = np.zeros(len(points))
    for idx,p in enumerate(points):
      line = Edge([minx,p[1]],p)
      count = np.sum(line.is_intersecting(self.edges))
      out[idx] = count%2

    return out.astype(bool)

  def __call__(self,t):
    return np.array([self.xinterp(t),self.yinterp(t)]).transpose()


    

