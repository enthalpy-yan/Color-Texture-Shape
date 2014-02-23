import numpy as np
import cv2
import sys
from scipy import sqrt, pi, arctan2
from scipy.ndimage import uniform_filter


#
# Helper functions
#

def thresholded(center, pixels):
    """
    Compare the center pixel to its 8 neighbors
    """
    ret = []
    for a in pixels:
        if a >= center:
            ret.append(1)
        else:
            ret.append(0)
    return ret

def getNeighborhood(img, idx, idy, default=0):
    """
    Given the center position to find the position of its neighbors.
    """
    try:
        return img[idx,idy]
    except IndexError:
        return default


def unique(a):
    """
    remove duplicates from input list
    """
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis = 0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis = 1)
    return a[ui]

def isValid(X, Y, point):
    """
    Check if point is a valid pixel
    """
    if point[0] < 0 or point[0] >= X:
        return False
    if point[1] < 0 or point[1] >= Y:
        return False
    return True
 
def getNeighbors(X, Y, x, y, dist):
    """
    Find pixel neighbors according to various distances
    """
    cn1 = (x + dist, y + dist)
    cn2 = (x + dist, y)
    cn3 = (x + dist, y - dist)
    cn4 = (x, y - dist)
    cn5 = (x - dist, y - dist)
    cn6 = (x - dist, y)
    cn7 = (x - dist, y + dist)
    cn8 = (x, y + dist)
 
    points = (cn1, cn2, cn3, cn4, cn5, cn6, cn7, cn8)
    Cn = []
 
    for i in points:
        if isValid(X, Y, i):
          Cn.append(i)

    return Cn
 
def correlogram(photo, Cm, K):
    """
    Get auto correlogram
    """
    X, Y, t = photo.shape
 
    colorsPercent = []

    for k in K:
        # print "k: ", k
        countColor = 0
 
        color = []
        for i in Cm:
           color.append(0)
 
        for x in range(0, X, int(round(X / 10))):
            for y in range(0, Y, int(round(Y / 10))):

                Ci = photo[x][y]
                Cn = getNeighbors(X, Y, x, y, k)
                for j in Cn:
                    Cj = photo[j[0]][j[1]]
 
                    for m in range(len(Cm)):
                        if np.array_equal(Cm[m], Ci) and np.array_equal(Cm[m], Cj):
                            countColor = countColor + 1
                            color[m] = color[m] + 1

        for i in range(len(color)):
            color[i] = float(color[i]) / countColor
        
        colorsPercent.append(color)

    return colorsPercent

def Resize(src,width,height):
    """
    Resize the img to given width and height
    """
    new = cv2.resize(src,(width,height))
    return new


#
# Image Descriptor functions
#

def autoCorrelogram(imgFile):
    """
    The functions for computing color correlogram. 
    To improve the performance, we consider to utilize 
    color quantization to reduce image into 64 colors. 
    So the K value of k-means should be 64.

    imgFile:
     source path of image. NOTE: shouldn't be the numpy ndarray.
    """
    img = cv2.imread(imgFile, 1)

    Z = img.reshape((-1, 3))

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

def lbp(img):
    """
    The function for computing Local binary patterns

    img: 
     The numpy ndarray that describe an image.
    """
    height, width = img.shape
    ret = np.empty([height, width], np.uint8)
    for x in range(0, len(img)):
        for y in range(0, len(img[0])):
            center        = img[x,y]
            top_left      = getNeighborhood(img, x-1, y-1)
            top_middle    = getNeighborhood(img, x, y-1)
            top_right     = getNeighborhood(img, x+1, y-1)
            right         = getNeighborhood(img, x+1, y )
            left          = getNeighborhood(img, x-1, y )
            bottom_left   = getNeighborhood(img, x-1, y+1)
            bottom_right  = getNeighborhood(img, x+1, y+1)
            bottom_middle = getNeighborhood(img, x,   y+1)

            values = thresholded(center, [top_left, top_middle, top_right, right, bottom_right,
                                          bottom_middle, bottom_left, left])

            weights = [1, 2, 4, 8, 16, 32, 64, 128]
            res = 0
            for a in range(0, len(values)):
                res += weights[a] * values[a]
            ret.itemset((x,y), res)
    return ret


def HoG(img, cell_per_blk=(3, 3), pix_per_cell=(8, 8), orientation=3):
    """
    The function for computing Histogram of oriented gradients. Takes 
    the following as keyword arguments:
    
    cell_per_blk:
     How may cells in a block ( cells in height, cells in width)

    pix_per_cell:
     The size of each cell: ( height, witdh )

    orientation:
     Binning the gradient into how many orientation bins
    """
    if img is None:
        print " pic read failed"
        return -1

    if img.ndim > 3:
        print " gray-scale process only for speed performance"
        return -1

    # gradient computation
    gradient_x = np.zeros(img.shape)
    gradient_y = np.zeros(img.shape)
    gradient_x[:, :-1] = np.diff(img, n = 1, axis = 1)
    gradient_y[:-1, :] = np.diff(img, n = 1, axis = 0)
    magnitude = sqrt(gradient_x ** 2 + gradient_y ** 2)
    ori = arctan2(gradient_y, (gradient_x + 1e-15)) * (180 / pi) + 90
    
    # Orientation Binning
    img_h, img_w = img.shape
    cx, cy = pix_per_cell
    bx, by = cell_per_blk
    ncell_x = int(np.floor(img_w // cx))
    ncell_y = int(np.floor(img_h // cy))
    ori_histogram = np.zeros((ncell_y, ncell_x, orientation))
    for i in range(0, orientation):
        temp1 = np.where(ori < 180 / orientation * (i + 1), ori, 0)
        temp1 = np.where(ori >= 180 / orientation * i, temp1, 0)
        temp2 = np.where(temp1>0, magnitude, 0)
        ori_histogram[:,:,i] = uniform_filter(temp2, size = (cy, cx))[cy / 2 :: cy, cx / 2 :: cx]

    # normalization
    n_blocksx = (ncell_x - bx) + 1
    n_blocksy = (ncell_y - by) + 1
    normalised_blocks = np.zeros((n_blocksy, n_blocksx, by, bx, orientation))
    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = ori_histogram[y : y + by, x : x + bx, :]
            eps = 1e-5
            normalised_blocks[y, x, :] = block / sqrt(block.sum() ** 2 + eps)

    return normalised_blocks

