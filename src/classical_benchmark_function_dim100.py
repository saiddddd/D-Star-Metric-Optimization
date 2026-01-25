#compile by : Said Al Afghani Edsa
# here, there are 24 functions to test the algorithm(s)

import numpy
import random
import math
import numpy as np

def prod( it ):
    p= 1
    for n in it:
        p *= n
    return p

def Ufun(x,a,k,m):
    y=k*((x-a)**m)*(x>a)+k*((-x-a)**m)*(x<(-a));
    return y

def fun_info(F):
    def F1(x):
        return sum([xi**2 for xi in x])

    def F2(x):
        return sum(abs(xi) for xi in x) + prod(abs(xi) for xi in x)

    def F3(x):
        dimension = len(x)
        R = 0
        for i in range(dimension):
            R += sum(x[:i+1])**2
        return R

    def F4(x):
        return max(abs(xi) for xi in x)

    def F5(x):
        dimension = len(x)
        return sum(100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(dimension-1))

    def F6(x):
        return sum(int(xi + 0.5)**2 for xi in x)

    def F7(x):
        dimension = len(x)
        return sum([(i+1)*(xi**4) for i, xi in enumerate(x)]) + random.random()

    def F8(x):
        return sum(-xi * math.sin(math.sqrt(abs(xi))) for xi in x)

    def F9(x):
        dim = len(x)
        o = np.sum(x**2 - 10 * np.cos(2 * np.pi * x)) + 10 * dim
        return o

    def F10(x):
        dimension = len(x)
        return -20 * math.exp(-0.2 * math.sqrt(sum(xi**2 for xi in x) / dimension)) - \
               math.exp(sum(math.cos(2 * math.pi * xi) for xi in x) / dimension) + 20 + math.e

    def F11(x):
        dim=len(x);
        w=[i for i in range(len(x))]
        w=[i+1 for i in w];
        o=numpy.sum(x**2)/4000-prod(numpy.cos(x/numpy.sqrt(w)))+1;   
        return o;

    def F12(x):
        dim=len(x);
        o=(math.pi/dim)*(10*((numpy.sin(math.pi*(1+(x[0]+1)/4)))**2)+numpy.sum((((x[1:dim-1]+1)/4)**2)*(1+10*((numpy.sin(math.pi*(1+(x[1:dim-1]+1)/4))))**2))+((x[dim-1]+1)/4)**2)+numpy.sum(Ufun(x,10,100,4));   
        return o;

    def F13(x): 
        dim=len(x);
        o=.1*((numpy.sin(3*math.pi*x[1]))**2+sum((x[0:dim-2]-1)**2*(1+(numpy.sin(3*math.pi*x[1:dim-1]))**2))+ 
        ((x[dim-1]-1)**2)*(1+(numpy.sin(2*math.pi*x[dim-1]))**2))+numpy.sum(Ufun(x,5,100,4));
        return o;

    if F == 'F1':
        fitness = F1
        lowerbound = -100
        upperbound = 100
        dimension = 100
    elif F == 'F2':
        fitness = F2
        lowerbound = -10
        upperbound = 10
        dimension = 100
    elif F == 'F3':
        fitness = F3
        lowerbound = -100
        upperbound = 100
        dimension = 100
    elif F == 'F4':
        fitness = F4
        lowerbound = -100
        upperbound = 100
        dimension = 100
    elif F == 'F5':
        fitness = F5
        lowerbound = -30
        upperbound = 30
        dimension = 100
    elif F == 'F6':
        fitness = F6
        lowerbound = -100
        upperbound = 100
        dimension = 100
    elif F == 'F7':
        fitness = F7
        lowerbound = -1.28
        upperbound = 1.28
        dimension = 100
    elif F == 'F8':
        fitness = F8
        lowerbound = -500
        upperbound = 500
        dimension = 100
    elif F == 'F9':
        fitness = F9
        lowerbound = -5.12
        upperbound = 5.12
        dimension = 100
    elif F == 'F10':
        fitness = F10
        lowerbound = -32
        upperbound = 32
        dimension = 100
    elif F == 'F11':
        fitness = F11
        lowerbound = -600
        upperbound = 600
        dimension = 100
    elif F == 'F12':
        fitness = F12
        lowerbound = -50
        upperbound = 50
        dimension = 100
    elif F == 'F13':
        fitness = F13
        lowerbound = -50
        upperbound = 50
        dimension = 100
        
    return lowerbound, upperbound, dimension, fitness
    
    
    