import numpy as np
import numpy 
import math 

def Get_Functions_details(F):
    def F1(x):
        return np.sum(x**2)
    
    def F2(x):
        return np.sum(np.abs(x)) + np.prod(np.abs(x))
    
    def F3(x):
        dim = len(x)
        o = 0
        for i in range(dim):
            o = o + np.sum(x[:i+1])**2
        return o
    
    def F4(x):
        return np.max(np.abs(x))
    
    def F5(x):
        dim = len(x)
        o = np.sum(100 * (x[1:dim] - (x[:dim-1]**2))**2 + (x[:dim-1] - 1)**2)
        return o
    
    def F6(x):
        return np.sum(np.abs(x + 0.5)**2)
    
    def F7(x):
        dim = len(x)
        o = np.sum(np.arange(1, dim+1) * (x**4)) + np.random.rand()
        return o
    
    def F8(x):
        return np.sum(-x * np.sin(np.sqrt(np.abs(x))))
    
    def F9(x):
        dim = len(x)
        o = np.sum(x**2 - 10 * np.cos(2 * np.pi * x)) + 10 * dim
        return o
    
    def F10(x):
        dim = len(x)
        o = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.exp(1)
        return o
    
    def F11(x):
        dim = len(x)
        o = np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, dim+1)))) + 1
        return o
    
    def F12(x):
        dim=len(x)
        o=(math.pi/dim)*(10*((numpy.sin(math.pi*(1+(x[0]+1)/4)))**2)+numpy.sum((((x[1:dim-1]+1)/4)**2)*(1+10*((numpy.sin(math.pi*(1+(x[1:dim-1]+1)/4))))**2))+((x[dim-1]+1)/4)**2)+numpy.sum(Ufun(x,10,100,4));   
        return o;
    
    def F13(x):
        dim = len(x)
        o = 0.1 * ((np.sin(3 * np.pi * x[0]))**2 + np.sum((x[:dim-1] - 1)**2 * (1 + (np.sin(3 * np.pi * x[1:dim]))**2)) + ((x[dim-1] - 1)**2) * (1 + (np.sin(2 * np.pi * x[dim-1]))**2)) + np.sum(Ufun(x, 5, 100, 4))
        return o

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
        aK = [.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246]
        bK = [.25, .5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
        aK = np.asarray(aK)
        bK = np.asarray(bK)
        bK = 1 / bK
        fit = np.sum((aK - ((L[0] * (bK**2 + L[1] * bK)) / (bK**2 + L[2] * bK)))**2)
        return fit
    
    def F16(L):  
        o=4*(L[0]**2)-2.1*(L[0]**4)+(L[0]**6)/3+L[0]*L[1]-4*(L[1]**2)+4*(L[1]**4);
        return o
    
    def F17(L):  
        o=(L[1]-(L[0]**2)*5.1/(4*(numpy.pi**2))+5/numpy.pi*L[0]-6)**2+10*(1-1/(8*numpy.pi))*numpy.cos(L[0])+10;
        return o
    
    def F18(x):
        o = (1 + (x[0] + x[1] + 1)**2 * (19 - 14 * x[0] + 3 * (x[0]**2) - 14 * x[1] + 6 * x[0] * x[1] + 3 * (x[1]**2))) * (30 + (2 * x[0] - 3 * x[1])**2 * (18 - 32 * x[0] + 12 * (x[0]**2) + 48 * x[1] - 36 * x[0] * x[1] + 27 * (x[1]**2)))
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
        aSH = [3, 5, 2, 1, 7]
        cSH = [2, 2, 4, 4, 4]

        o = 0
        for i in range(5):
            #o = o - (np.dot(x - aSH[i], x - aSH[i]) + cSH[i])**(-1)
   
            o = o - 1.0 / (np.dot(x - aSH[i], (x - aSH[i]).T) + cSH[i] + 1e-15)

        return o

    def F22(L):
        aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
        cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

        fit = 0
        for i in range(len(aSH)):
            try:
                v = np.matrix(np.subtract(L, aSH[i][:, np.newaxis]))
                inverse = np.linalg.inv(v * v.T + cSH[i])
                fit = fit - inverse.item((0, 0))
            except np.linalg.LinAlgError:
                continue

        return fit

    def F23(x):
        if len(x) < 4:
            raise ValueError("Parameter x harus memiliki panjang setidaknya 4 untuk fungsi F23")

        aSH = [0.1, 0.2, 0.2, 0.4]
        cSH = [0.1, 0.2, 0.2, 0.4]

        fit = np.sum(np.square(x[:4] - aSH))

        for i in range(4, len(aSH)):
            try:
                v = np.matrix(np.subtract(x[:3], aSH[i][:3]))  # Menggunakan hanya 3 parameter pertama dari aSH
                inverse = np.linalg.inv(v * v.T + cSH[i])
                fit = fit - inverse.item((0, 0))
            except:
                fit = fit

        return fit


    
    def Ufun(x, a, k, m):
        return k * ((x - a)**m) * (x > a) + k * ((-x - a)**m) * (x < -a)

    lb, ub, dim, fobj = None, None, None, None
    
    if F == 'F1':
        fobj = F1
        dim = 30
        lb = np.ones(dim) * -100
        ub = np.ones(dim) * 100
    elif F == 'F2':
        fobj = F2
        dim = 30
        lb = np.ones(dim) * -10
        ub = np.ones(dim) * 10
    elif F == 'F3':
        fobj = F3
        dim = 30                                           
        lb = np.ones(dim) * -100
        ub = np.ones(dim) * 100
    elif F == 'F4':
        fobj = F4
        dim = 30
        lb = np.ones(dim) * -100
        ub = np.ones(dim) * 100
    elif F == 'F5':
        fobj = F5
        dim = 30
        lb = np.ones(dim) * -30
        ub = np.ones(dim) * 30
    elif F == 'F6':
        fobj = F6
        dim = 30
        lb = np.ones(dim) * -100
        ub = np.ones(dim) * 100
    elif F == 'F7':
        fobj = F7
        dim = 30
        lb = np.ones(dim) * -1.28
        ub = np.ones(dim) * 1.28
    elif F == 'F8':
        fobj = F8
        dim = 30
        lb = np.ones(dim) * -500
        ub = np.ones(dim) * 500
    elif F == 'F9':
        fobj = F9
        dim = 30
        lb = np.ones(dim) * -5.12
        ub = np.ones(dim) * 5.12
    elif F == 'F10':
        fobj = F10
        dim = 30
        lb = np.ones(dim) * -32
        ub = np.ones(dim) * 32
    elif F == 'F11':
        fobj = F11
        dim = 30
        lb = np.ones(dim) * -600
        ub = np.ones(dim) * 600
    elif F == 'F12':
        fobj = F12
        dim = 30
        lb = np.ones(dim) * -50
        ub = np.ones(dim) * 50
    elif F == 'F13':
        fobj = F13
        dim = 30
        lb = np.ones(dim) * -50
        ub = np.ones(dim) * 50
    elif F == 'F14':
        fobj = F14
        dim = 2
        lb = np.ones(dim) * -65.536
        ub = np.ones(dim) * 65.536
    elif F == 'F15':
        fobj = F15
        dim = 4
        lb = np.ones(dim) * -5
        ub = np.ones(dim) * 5
    elif F == 'F16':
        fobj = F16
        dim = 2
        lb = np.ones(dim) * -5
        ub = np.ones(dim) * 5
    elif F == 'F17':
        fobj = F17
        dim = 2
        lb = np.array([-5, 0])
        ub = np.array([10, 15])
    elif F == 'F18':
        fobj = F18
        dim = 2
        lb = np.ones(dim) * -2
        ub = np.ones(dim) * 2
    elif F == 'F19':
        fobj = F19
        dim = 3
        lb = np.zeros(dim)
        ub = np.ones(dim)
    elif F == 'F20':
        fobj = F20
        dim = 6
        lb = np.zeros(dim)
        ub = np.ones(dim)
    elif F == 'F21':
        fobj = F21
        dim = 4
        lb = np.zeros(dim)
        ub = np.array([0, 10, 0, 10])
    elif F == 'F22':
        fobj = F22
        dim = 4
        lb = np.zeros(dim)
        ub = np.array([0, 10, 0, 10])
    elif F == 'F23':
        fobj = F23
        dim = 4
        lb = np.zeros(dim)
        ub = np.array([0, 10, 0, 10])
    
    return lb, ub, dim, fobj
