#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from misc import Timer
timer = Timer()

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

  def is_parallel(self,edge):
    '''
    returns True when edge is parallel to self
    '''
    a = np.cross(edge.dv,self.dv)
    if a == 0:
      return True

    return False

  def is_collinear(self,edge):
    '''
    returns True when edge is collinear with self
    '''
    if self.is_parallel(edge):
      a = np.cross(edge.v1-self.v1,self.dv)             
      if a == 0:
        return True

    return False

  def is_overlapping(self,edge):
    '''
    returns True if there is a finite width of overlap between edge and self.
    Overlapping vertices do not count as an overlap
    '''
    if self.is_collinear(edge):
      a = np.dot(edge.v1-self.v1,self.dv)
      b = np.dot(self.v1-edge.v1,edge.dv)
      c = np.dot(self.dv,self.dv)
      d = np.dot(edge.dv,edge.dv)
      if (a > 0) & (a < c):
        return True

      if (b > 0) & (b < d):
        return True

      return False

  def is_intersecting(self,edge):
    '''
    returns True if there is any point intersection between edge and self.
    Overlapping vertices does not count as an intersection.
    '''
    if self.is_parallel(edge):
      return False

    a = np.cross(edge.v1-self.v1,edge.dv)
    b = np.cross(self.v1-edge.v1,self.dv)    
    c = np.cross(self.dv,edge.dv)
    d = np.cross(edge.dv,self.dv) 
    t = a/c
    u = b/d
    if ((t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)):
      return True

    return False

  def intersection(self,edge):
    '''
    returns the point of intersection between edge and self if it exists
    '''
    if self.is_parallel(edge):
      raise TopologyError('intersection point does not exist')

    else:
      a = np.cross(edge.v1-self.v1,edge.dv)
      b = np.cross(self.v1-edge.v1,self.dv)    
      c = np.cross(self.dv,edge.dv)
      d = np.cross(edge.dv,self.dv) 
      t = a/c
      u = b/d
      return self.v1 + t*self.dv

class JordanCurve(object):
  def __init__(self,points):
    assert len(np.shape(points)) == 2
    assert np.shape(points)[1] == 2
    N = np.shape(points)[0]
    self.edges = []
    for i in range(1,N):
      e = Edge(points[i-1,:],points[i,:])
      if i > 1:
        if any(e.is_overlapping(f) for f in self.edges):
          raise TopologyError('creating an edge with points %s and %s causes '
                              'an overlap' % (points[i-1],points[i]))

      if (i > 2):
        if any(e.is_intersecting(f) for f in self.edges[:-1]):
          raise TopologyError('creating an edge with points %s and %s causes '
                              'an intersection' % (points[i-1],points[i]))

      self.edges += [e]

    e = Edge(points[-1,:],points[0,:])
    if any(e.is_overlapping(f) for f in self.edges):
      raise TopologyError('creating an edge with points %s and %s causes an '
                          'overlap' % (points[-1],points[0]))

    if any(e.is_intersecting(f) for f in self.edges[1:-1]):
      raise TopologyError('creating an edge with points %s and %s causes an '
                          'intersection' % (points[-1],points[0]))

    self.edges += [e]

    self.x = np.array([e.x for e in self.edges])
    self.y = np.array([e.y for e in self.edges])
  

D = JordanCurve(points)
plt.plot(D.x,D.y,'ko-')
plt.show()
