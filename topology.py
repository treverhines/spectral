#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from misc import Timer
timer = Timer()

def io_check(fin):
  def fout(self,edge):
    is_edge = isinstance(edge,Edge)
    if is_edge:
      edge = [edge]

    out = fin(self,edge)
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
  def __init__(self,v1,v2):
    self.v1 = np.asarray(v1)
    self.v2 = np.asarray(v2)
    self.dv = self.v2 - self.v1
    self.x = np.array([v1[0],v2[0]])
    self.y = np.array([v1[1],v2[1]])

  @io_check
  def is_parallel(self,edge):
    '''
    returns True when edge is parallel to self
    '''
    dv_list = np.array([e.dv for e in edge]) 
    a = np.cross(dv_list,self.dv)
    cond1 = (a == 0)          
    return cond1

  @io_check
  def is_collinear(self,edge):
    '''
    returns True when edge is collinear with self
    '''
    v1_list = np.array([e.v1 for e in edge]) 
    a = np.cross(v1_list-self.v1,self.dv)             
    cond1 = self.is_parallel(edge)
    cond2 = (a == 0)
    return cond1 & cond2

  @io_check
  def is_overlapping(self,edge):
    '''
    returns True if there is a finite width of overlap between edge and self.
    Overlapping vertices do not count as an overlap
    '''
    v1_list = np.array([e.v1 for e in edge]) 
    dv_list = np.array([e.dv for e in edge]) 
    a = np.sum((v1_list-self.v1)*self.dv,1)
    b = np.sum((self.v1-v1_list)*dv_list,1)
    c = np.dot(self.dv,self.dv)
    d = np.sum(dv_list*dv_list,1)
    cond1 = self.is_collinear(edge)
    cond2 = (a > 0) & (a < c)
    cond3 = (b > 0) & (b < d)
    return cond1 & (cond2 | cond3)
      

  @io_check
  def is_intersecting(self,edge):
    '''
    returns True if there is any point intersection between edge and self.
    Overlapping vertices does not count as an intersection.
    '''
    v1_list = np.array([e.v1 for e in edge]) 
    dv_list = np.array([e.dv for e in edge]) 

    cond1 = self.is_parallel(edge) == False
    v1_list = v1_list[cond1]
    dv_list = dv_list[cond1]

    a = np.cross(v1_list-self.v1,dv_list)
    b = np.cross(self.v1-v1_list,self.dv)
    c = np.cross(self.dv,dv_list)
    t = a/c
    u = -b/c
    cond2 = (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    cond1[cond1] = cond2
    return cond1

  @io_check
  def intersection(self,edge):
    '''
    returns the point of intersection between edge and self if it exists
    '''
    v1_list = np.array([e.v1 for e in edge]) 
    dv_list = np.array([e.dv for e in edge]) 

    if any(self.is_intersecting(edge) == False):
      raise TopologyError('an intersection point does not exist')

    a = np.cross(v1_list-self.v1,dv_list)
    b = np.cross(self.v1-v1_list,self.dv)    
    c = np.cross(self.dv,dv_list)
    d = np.cross(dv_list,self.dv) 
    t = a/c
    u = b/d
    return self.v1 + t[:,None]*self.dv

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

    self.x = np.array([e.x[0] for e in self.edges]+[self.edges[0].x[0]])
    self.y = np.array([e.y[0] for e in self.edges]+[self.edges[0].y[0]])

  def contains(self,point):
    line = Edge([min(self.x),point[1]],point)
    count = np.sum(line.is_intersecting(self.edges))
    return bool(count%2)


    

