import numpy as np

def CosineSimilarity(i1, i2):
    """
    Function for calculating cosine similarity between two image
    
    i1, i2:
     numpy ndarray for image
     
    """
    numerator = (i1 * i2).sum()
    denoma = (i1 * i1).sum()
    denomb = (i2 * i2).sum()
    return 1 - numerator / np.sqrt(denoma*denomb)

def hist_intersection(A, B):
    """
    Function for calculating histogram intersection between two image
    
    A, B:
     numpy ndarray for image
    """
    min_matrix = np.where(A >= B, B, 0) + np.where(A < B, A, 0)
    the_min = min_matrix / float(min(np.sum(A.ravel()), np.sum(B.ravel())))
    return sum(the_min.ravel())
