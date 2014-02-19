import numpy as np

def L1Dist(i1, i2):
    return np.sum(np.abs(img1 - img2))

def L2Dist(i1, i2):
    return np.sqrt(np.sum((i1 - i2)**2))
