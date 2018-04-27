import torch as t
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from numba import jit
import time
import functools

def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
    return newfunc

@jit(nopython=True)
def l2_dist(current_p):
    n,_ = current_p.shape
    dist = np.zeros(n*n)
    for j in range(0, n):
            p = current_p[j] #* np.ones((n,3)) #broadcasting is faster
            d = (p - current_p)*(p - current_p) 
            #d = np.sqrt(d.sum(1)) # cannot use 'sum()' in nopython mode
            dsum = np.sqrt(d[:,0] + d[:,1] + d[:,2])  
            dist[j:j+n] = dsum 
    return dist

def dist_ball_point2(radius,nsample,xyz, nn_k):
    dist_all = []
    b,n,_ = xyz.shape
    #num_elem = n*n-1
    for i in range(0, b):
        print(i)
        #dist = []
        current_p = xyz[i].data.numpy()
        dist = l2_dist(current_p)
        #for j in range(0, n):
        #    cnt = 0
        #    p = current_p[j]
        #    #print(xyz[i].data.numpy())
        #    d = (p - current_p)*(p - current_p)
        #    d = np.sqrt(d.sum(1))
        #    dist.extend(d)
        dist = np.array(dist)
        dist[dist<radius] = 0
        dist_topk = dist[np.argpartition(dist, -nn_k)[-nn_k:]]  # top k
        dist_all.append(dist_topk)
    return np.concatenate(dist_all)



@timeit
def get_repulsion_loss5(points, nsample=20, radius=0.07):
    dist_all = dist_ball_point2(radius, nsample, points,5)
    h = 0.3
    weight = np.exp(-dist_all / h ** 2)
    uniform_loss = (radius - dist_all * weight).sum()
    return uniform_loss




t.manual_seed(21)
p = t.rand(32,1024,3) 
points = Variable(p) 

loss = get_repulsion_loss5(points) 
print(loss)

