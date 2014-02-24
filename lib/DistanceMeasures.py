import numpy as np

def L1Dist(i1, i2):
    """
    L1 Distance
    """
    return np.sum(np.abs(i1 - i2))

def L2Dist(i1, i2):
    """
    L2 Distance
    """
    return np.sqrt(np.sum((i1 - i2) ** 2))

def ChiSquareDist(i1, i2):
    """
    Chi-Square Distance
    """
    return np.sum(np.where(i1 > 0, ((i1 - i2) ** 2) / i1, 0))

def KLDist(i1, i2):
    """
    KL Distance
    """
    return np.sum(np.where(i1 != 0, i1 * np.log(i1 / i2), 0))
