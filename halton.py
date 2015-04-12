#!/usr/bin/env python
from __future__ import division
import numpy as np

class Prime(object):
  '''
  enumerates over N prime numbers
  '''  
  def __init__(self,N):
    self.N = N
    self.n = 0

  def __iter__(self):
    return self

  def next(self):
    if self.n == self.N:
      raise StopIteration

    elif self.n == 0:
      self.primes = [2]
      self.n += 1


    else:
      test = self.primes[-1] + 1
      while True:
        if not any(test/p == test//p for p in self.primes):
          self.primes += [test]  
          self.n += 1
          break 

        test += 1

    out = self.n-1,self.primes[-1]
    return out

def halton(N,dim=1):
  '''
  returns a halton sequence using the algorithm 
  from http://en.wikipedia.org/wiki/Halton_sequence
  '''
  out = np.zeros((N,dim))
  for d,base in Prime(dim):
    i = np.arange(1,N+1)
    f = np.ones(N)
    result = np.zeros(N)
    while any(i > 0):
      f = f/base
      result += f*(i%base)
      i //= base 
    
    out[:,d] = result

  return out

def qunif(N,low=0.0,high=1.0):
  '''
  quasi-uniform random number generator
  '''
  assert np.shape(low) == np.shape(high)
  assert len(np.shape(low)) <= 1
  if len(np.shape(low)) == 1:
    low = np.asarray(low)
    high = np.asarray(high)
    dim = len(low)
    shape = (N,dim)

  elif len(np.shape(low)) == 0:
    dim = 1
    shape = (N,)

  out = (halton(N,dim)*(high-low) + low)
  out = np.reshape(out,shape)
  return out  
  
def qnorm(N,mu=0.0,sigma=1.0):  
  ''' 
  quasi-normal random number generator
  '''
  assert np.shape(mu) == np.shape(sigma)
  assert len(np.shape(mu)) <= 1
  if len(np.shape(mu)) == 1:
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    dim = len(mu)
    shape = (N,dim)

  elif len(np.shape(mu)) == 0:
    dim = 1
    shape = (N,)

  unif = halton(N,2*dim)
  unif1 = unif[:,:dim]
  unif2 = unif[:,dim:]
  out = np.sqrt(-2*np.log(unif1))*np.cos(2*np.pi*unif2)
  out = (out*sigma + mu)
  out = np.reshape(out,shape)
  return out  

      
    
  







    
    
