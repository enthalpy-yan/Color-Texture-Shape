import numpy as np

#
# L1 Distance
#
def L1Dist(i1, i2):
    return np.sum(np.abs(img1 - img2))

#
# L2 Distance
#
def L2Dist(i1, i2):
    return np.sqrt(np.sum((i1 - i2) ** 2))

#
# Chi-Square Distance
#
def ChiSquareDist(i1, i2):
    return np.sum(2 * ((i1 - i2) ** 2) / (i1 + i2))

#
# KL Distance
#
def KLDist(i1, i2):
    return np.sum(np.where(i1 != 0, i1 * np.log(i1 / i2), 0))
