import cv2
import sys
import numpy as np

def unique(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]

# check if point is a valid pixel
def is_valid(X, Y, point):
    if point[0]<0 or point[0]>=X:
        return False
    if point[1]<0 or point[1]>=Y:
        return False
    return True
 
# find pixel neighbors
def get_neighbors(X, Y, x, y, dist):
    cn1 = (x+dist, y+dist)
    cn2 = (x+dist, y)
    cn3 = (x+dist, y-dist)
    cn4 = (x, y-dist)
    cn5 = (x-dist, y-dist)
    cn6 = (x-dist, y)
    cn7 = (x-dist, y+dist)
    cn8 = (x, y+dist)
 
    points = (cn1, cn2, cn3, cn4, cn5, cn6, cn7, cn8)
    Cn = []
 
    for i in points:
        if is_valid(X, Y, i):
          Cn.append(i)
 
    return Cn
 
# get correlogram
def correlogram(photo, Cm, K):

    X, Y, ttt = photo.shape
    # X, Y = photo.shape
 
    colors_percent = []
 
    for k in K:
        print "k: ", k
        countColor = 0
 
        color = []
        for i in Cm:
           color.append(0)
 
        for x in range(0, X, int(round(X/10))):
            for y in range(0, Y, int(round(Y/10))):

                Ci = photo[x][y]
                Cn = get_neighbors(X, Y, x, y, k)
                for j in Cn:
                    Cj = photo[j[0]][j[1]]
 
                    for m in range(len(Cm)):
                        # print "Cm ", m, " : ", Cm[m]
                        # print "Ci ", " : ", Ci, " ", "Cj ", " : ", Cj
                        if np.array_equal(Cm[m], Ci) and np.array_equal(Cm[m], Cj):
                            countColor = countColor + 1
                            color[m] = color[m] + 1
                            # print "same color"
        print countColor
        for i in range(len(color)):
            color[i] = float(color[i]) / countColor
        
        colors_percent.append(color)

    return colors_percent

def autoCorrelogram(imgFile):
    img = cv2.imread(imgFile, 1)

    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 64
    ret, label, center = cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # according to "Image Indexing Using Color Correlograms" paper
    K = [i for i in range(1, 9, 2)]

    colors64 = unique(np.array(res))

    result = correlogram(res2, colors64, K)
    return result