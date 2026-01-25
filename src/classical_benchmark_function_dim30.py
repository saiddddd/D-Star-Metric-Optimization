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

    def F14(x):
        aS = np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
                       [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
        bS = np.zeros(25)
        v = np.matrix(x).reshape(-1, 1)  # Mengubah dimensi x menjadi (13, 1)

        for i in range(25):
            H = v - aS[:, i]
            bS[i] = np.sum((np.power(H, 6)))

        w = np.arange(1, 26)
        o = ((1.0 / 500) + np.sum(1.0 / (w + bS))) ** (-1)
        return o

    def F15(L):  
        aK=[.1957,.1947,.1735,.16,.0844,.0627,.0456,.0342,.0323,.0235,.0246];
        bK=[.25,.5,1,2,4,6,8,10,12,14,16];
        aK=numpy.asarray(aK);
        bK=numpy.asarray(bK);
        bK = 1/bK;  
        fit=numpy.sum((aK-((L[0]*(bK**2+L[1]*bK))/(bK**2+L[2]*bK+L[3])))**2);
        return fit
    
    def F16(L):  
        o=4*(L[0]**2)-2.1*(L[0]**4)+(L[0]**6)/3+L[0]*L[1]-4*(L[1]**2)+4*(L[1]**4);
        return o


    def F17(L):
        o = (L[1] - (L[0] ** 2) * 5.1 / (4 * (np.pi ** 2)) + 5 / np.pi * L[0] - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(L[0]) + 10
        return o


    def F18(L):  
        o=(1+(L[0]+L[1]+1)**2*(19-14*L[0]+3*(L[0]**2)-14*L[1]+6*L[0]*L[1]+3*L[1]**2))*(30+(2*L[0]-3*L[1])**2*(18-32*L[0]+12*(L[0]**2)+48*L[1]-36*L[0]*L[1]+27*(L[1]**2)));
        return o
    
    def F19(L):
        aH = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
        cH = np.array([1, 1.2, 3, 3.2])
        pH = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
        o = 0

        for i in range(4):
            o = o - cH[i] * np.exp(-np.sum(aH[i, :] * ((L[:, np.newaxis] - pH[i, :]) ** 2)))

        return o

    def F20(L):    
        aH = [[10, 3, 17, 3.5, 1.7, 8],
              [0.05, 10, 17, 0.1, 8, 14],
              [3, 3.5, 1.7, 10, 17, 8],
              [17, 8, 0.05, 10, 0.1, 14]]
        aH = np.asarray(aH)
        cH = [1, 1.2, 3, 3.2]
        cH = np.asarray(cH)
        pH = [[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
              [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
              [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
              [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]
        pH = np.asarray(pH)
        o = 0
        for i in range(0, 4):
            o = o - cH[i] * np.exp(-np.sum(aH[i, :] * ((L[:6] - pH[i, :6]) ** 2)))
        return o

    def F21(x):
        aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
        cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

        o = 0.0
        for i in range(5):
            v = np.subtract(x, aSH[i])
            try:
                inverse = 1.0 / (np.dot(v, v.T) + cSH[i])
                o = o - inverse
            except np.linalg.LinAlgError:
                continue

        return o

    def F22(x):
        aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
        cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

        o = 0.0
        for i in range(7):
            v = np.subtract(x, aSH[i])
            try:
                inverse = 1.0 / (np.dot(v, v.T) + cSH[i])
                o = o - inverse
            except np.linalg.LinAlgError:
                continue

        return o

    def F23(x):
        aSH = [[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
               [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]]
        cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
        R = 0
        for i in range(10):
            R -= ((x - aSH[i]) @ (x - aSH[i]).T + cSH[i])**(-1)
        return R
    
    def F24(x):
        new_x = []
#         for i in range(0, len(x)):
#             new_x.append(i*x[i]**2)
        
        for i in range(0, len(x)):
            new_x.append(x[i] - i)
        output = sum([xi**2 for xi in new_x])
        return output

    if F == 'F1':
        fitness = F1
        lowerbound = -100
        upperbound = 100
        dimension = 30
    elif F == 'F2':
        fitness = F2
        lowerbound = -10
        upperbound = 10
        dimension = 30
    elif F == 'F3':
        fitness = F3
        lowerbound = -100
        upperbound = 100
        dimension = 30
    elif F == 'F4':
        fitness = F4
        lowerbound = -100
        upperbound = 100
        dimension = 30
    elif F == 'F5':
        fitness = F5
        lowerbound = -30
        upperbound = 30
        dimension = 30
    elif F == 'F6':
        fitness = F6
        lowerbound = -100
        upperbound = 100
        dimension = 30
    elif F == 'F7':
        fitness = F7
        lowerbound = -1.28
        upperbound = 1.28
        dimension = 30
    elif F == 'F8':
        fitness = F8
        lowerbound = -500
        upperbound = 500
        dimension = 30
    elif F == 'F9':
        fitness = F9
        lowerbound = -5.12
        upperbound = 5.12
        dimension = 30
    elif F == 'F10':
        fitness = F10
        lowerbound = -32
        upperbound = 32
        dimension = 30
    elif F == 'F11':
        fitness = F11
        lowerbound = -600
        upperbound = 600
        dimension = 30
    elif F == 'F12':
        fitness = F12
        lowerbound = -50
        upperbound = 50
        dimension = 30
    elif F == 'F13':
        fitness = F13
        lowerbound = -50
        upperbound = 50
        dimension = 30
    elif F == 'F14':
        fitness = F14
        lowerbound = -65.536
        upperbound = 65.536
        dimension = 2
    elif F == 'F15':
        fitness = F15
        lowerbound = -5
        upperbound = 5
        dimension = 4
    elif F == 'F16':
        fitness = F16
        lowerbound = -5
        upperbound = 5
        dimension = 2
    elif F == 'F17':
        fitness = F17
        lowerbound = -5
        upperbound = 5
        dimension = 2
    elif F == 'F18':
        fitness = F18
        lowerbound = -2
        upperbound = 2
        dimension = 2
    elif F == 'F19':
        fitness = F19
        lowerbound = 0
        upperbound = 1
        dimension = 3
    elif F == 'F20':
        fitness = F20
        lowerbound = 0
        upperbound = 1
        dimension = 6
    elif F == 'F21':
        fitness = F21
        lowerbound = 0
        upperbound = 10
        dimension = 4
    elif F == 'F22':
        fitness = F22
        lowerbound = 0
        upperbound = 10
        dimension = 4
    elif F == 'F23':
        fitness = F23
        lowerbound = 0
        upperbound = 10
        dimension = 4
    elif F == 'F24':
        fitness = F24
        lowerbound = -100 #-10
        upperbound = 100 #10
        dimension = 30 #30

    return lowerbound, upperbound, dimension, fitness
    
    
    