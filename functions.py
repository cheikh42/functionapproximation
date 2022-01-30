from math import exp
import numpy as np


def f0(x):
    return x**2


def f1(x, y):
    return x*x - y*y

def f2(x,y):
    return x*y * np.exp( -( x*x + y*y ) )

def f3(x):
    return np.sin( x )

## This function returns the values of the function adapted to a numpy array.
## So if the input is X = [ x1 = [a, b, c],x2 = [a, b1, c1],x3 = [a2, b2, c2] ] 
## This will give us f(X)=[f(x1), f(x2), f(x3)]
def f4(x):
    return np.exp(np.array([-np.linalg.norm(i-2)**2 for i in x]) / (2/5) )


## This function plots the function in 3d and is adapted to np.meshgrid. 
## It serves only for the purpose of visualization.
def f4_3D(x,y):
    return np.exp( - ( (x-2)**2 + (y-2)**2 )  / (2/5) ) 


def f5(x,y):
    return np.sin(x)+np.sin(y)

