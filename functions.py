import numpy as np


def f0(x):
    return x**2


def f1(x, y):
    return x*x - y*y

def f2(x,y):
    return x*y * np.exp( -( x*x + y*y ) )

def f3(x):
    return np.sin( x )

def f4(x):
    return np.exp( -np.linalg.norm(x-2) / (2/5) )