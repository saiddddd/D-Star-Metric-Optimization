import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Get_Functions_details import *

def func_plot(func_name, ax):
    lb, ub, dim, fobj = Get_Functions_details(func_name)

    if func_name in ['F1', 'F2', 'F3', 'F4', 'F6', 'F14']:
        x = np.arange(-100, 101, 2)
        y = x  # Tambahkan inisialisasi y di sini
    elif func_name == 'F5':
        x = np.arange(-200, 201, 2)
        y = x  # Tambahkan inisialisasi y di sini
    elif func_name in ['F7', 'F16']:
        x = np.arange(-1, 1.03, 0.03)
        y = x  # Tambahkan inisialisasi y di sini
    elif func_name == 'F8':
        x = np.arange(-500, 501, 10)
        y = x  # Tambahkan inisialisasi y di sini
    elif func_name == 'F9':
        x = np.arange(-5, 5.1, 0.1)
        y = x  # Tambahkan inisialisasi y di sini
    elif func_name == 'F10':
        x = np.arange(-20, 20.5, 0.5)
        y = x  # Tambahkan inisialisasi y di sini
    elif func_name == 'F11':
        x = np.arange(-500, 501, 10)
        y = x  # Tambahkan inisialisasi y di sini
    elif func_name == 'F12':
        x = np.arange(-10, 10.1, 0.1)
        y = x  # Tambahkan inisialisasi y di sini
    elif func_name == 'F13':
        x = np.arange(-5, 5.08, 0.08)
        y = x  # Tambahkan inisialisasi y di sini
    elif func_name == 'F15':
        x = np.arange(-5, 5.1, 0.1)
        y = x  # Tambahkan inisialisasi y di sini
    elif func_name == 'F17':
        x = np.arange(-5, 10.1, 0.1)
        y = np.arange(0, 15.1, 0.1)
    elif func_name == 'F18':
        x = np.arange(-2, 2.1, 0.1)
        y = np.arange(-2, 2.1, 0.1)
    elif func_name in ['F19', 'F20', 'F21', 'F22', 'F23']:
        x = np.arange(-5, 5.1, 0.1)
        y = x  # Tambahkan inisialisasi y di sini

    L = len(x)
    f = np.zeros((L, L))

    for i in range(L):
        for j in range(L):
            if func_name not in ['F15', 'F19', 'F20', 'F21', 'F22', 'F23']:
                f[i, j] = fobj(np.array([x[i], y[j], 0]))  # Tambahkan 0 jika diperlukan
            elif func_name == 'F15':
                f[i, j] = fobj(np.array([x[i], y[j], 0, 0]))  # Tambahkan 0 jika diperlukan
            elif func_name == 'F19':
                f[i, j] = fobj(np.array([x[i], y[j], 0, 0]))  # Sesuaikan jumlah parameter
            elif func_name == 'F20':
                f[i, j] = fobj(np.array([x[i], y[j], 0, 0, 0, 0]))  # Sesuaikan panjang input
            elif func_name in ['F21', 'F22', 'F23']:
                f[i, j] = fobj(np.array([x[i], y[j], 0]))  # Tambahkan 0 jika diperlukan

    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, f, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('F')