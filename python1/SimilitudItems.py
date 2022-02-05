import numpy as np
import pandas as pd
import math
from numba import jit

# @jit(nopython=True)
# @jit
def SimilitudItems(data, method):
    length = data.shape[1]
    D = np.zeros((length, length))
    if(str.lower(method) == "cosine"):
        for i in range(0, length - 1):
            for j in range(i + 1, length):
                D[i, j] = sum(np.multiply(data[:, i], data[:, j])) / (np.linalg.norm(data[:, i], 2) * np.linalg.norm(data[:, j], 2))
                if (D[i, j] == None):
                    D[i, j] = 0
                D[i,j] = abs(D[i, j])
    elif(str.lower(method) == "correlation"):
        for i in range(0, length - 1):
            for j in range(i + 1, length):
                temp = find(data[:, i] != 0 and data[:, j] != 0)
                Rui = data[temp, i]
                Ruj = data[temp, j]
                Ri = mean(data[:, i])
                Rj = mean(data[:, j])
                D[i, j] = sum(np.multiply((Rui - Ri).T, Ruj - Rj)) / (np.linalg.norm(Rui - Ri) * np.linalg.norm(Ruj - Rj))
                if (D[i, j] == None):
                    D[i, j] = 0
                D[i, j] = abs(D[i, j])
    elif(str.lower(method) == "adjustedcosine"):
        for i in range(0, length - 1):
            for j in range(i + 1, length):
                temp = find(data[:, i] != 0 and data[:,j] != 0)
                Rui = data[temp, i]
                Ruj = data[temp, j]
                Ru = mean(data[temp, :].T).T
                D[i,j] = sum(np.multiply((Rui - Ru).T, Ruj - Ru)) / (np.linalg.norm(Rui - Ru) * np.linalg.norm(Ruj - Ru))
                if (D[i, j] == None):
                    D[i, j] = 0
                D[i, j] = abs(D[i, j])
    D = D.T + D
    return D