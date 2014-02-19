import numpy as np
import cv2
import DistanceMeasures as dm

#
# Compare the center pixel to its 8 neighbors
#
def thresholded(center, pixels):
    ret = []
    for a in pixels:
        if a >= center:
            ret.append(1)
        else:
            ret.append(0)
    return ret

#
# Given the center position to find the position of its neighbors.
#
def getNeighborhood(img, idx, idy, default=0):
    try:
        return img[idx,idy]
    except IndexError:
        return default

#
# Compute LBP feature vector
#
def basic_lbp(img):
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

#
# Test case
#
img = cv2.imread('lena.jpg', 0)
transformed_img = basic_lbp(img)

print dm.L2Dist(img, img)

cv2.imshow('image', img)
cv2.imshow('thresholded image', transformed_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
