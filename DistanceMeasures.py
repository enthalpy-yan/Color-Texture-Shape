import numpy as np

def L1Dist(i1, i2):
    return np.sum(np.abs(img1 - img2))

def L2Dist(i1, i2):
    return np.sqrt(np.sum((i1 - i2) ** 2))

def ChiSquareDist(i1, i2):
    return np.sum(2 * ((i1 - i2) ** 2) / (i1 + i2))

def KLDist(i1, i2):
    return np.sum(np.where(i1 != 0, i1 * np.log(i1 / i2), 0))
