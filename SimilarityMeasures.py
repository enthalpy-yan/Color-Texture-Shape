import numpy as np

def CosineSimilarity(i1, i2):
    numerator = (i1 * i2).sum()
    denoma = (i1 * i1).sum()
    denomb = (i2 * i2).sum()
    return 1 - numerator / np.sqrt(denoma*denomb)
