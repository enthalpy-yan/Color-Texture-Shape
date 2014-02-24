import sys
import numpy as np
import cv2
import lib.Descriptors as des
import lib.DistanceMeasures as dm
import lib.SimilarityMeasures as sm

if __name__ == "__main__":
    print "Test start..."
    img1 = cv2.imread('lena.jpg', 0)
    img2 = cv2.imread('lena.jpg', 0)
    img4AutoCorr = cv2.imread('lena.jpg', 1)
    
    print "Testing Color Correlogram"
    matrix = des.autoCorrelogram(img4AutoCorr)
    for i in range(0, 4): 
        print "k = ", 2 * i + 1
        print matrix[i]
    
    print "Testing LBP"
    transformed_img = des.lbp(img1)
    cv2.imshow('image', img1)
    cv2.imshow('thresholded image', transformed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print "Testing HOG"
    hog = des.HoG(img1,See_graph=True)
    print "The output Matrix of HOG: "
    print hog

    print "Testing distance calculation functions"
    print "L1 distance is " + str(dm.L1Dist(img1, img2))
    print "L2 distance is " + str(dm.L2Dist(img1, img2))
    print "Chi-square distance is " + str(dm.ChiSquareDist(img1, img2))
    print "KL distance is " + str(dm.KLDist(img1, img2))

    print "Testing similarity functions"
    img3 = cv2.imread('bunny.jpg', 0)
    img3 = des.Resize(img3,img1.shape[0],img1.shape[1])
    print "Cosine similarity of the same img is " + str(sm.CosineSimilarity(img1, img2))
    print "Cosine similarity of the different img is" + str(sm.CosineSimilarity(img1, img3))
    print "Histogram Intersection similarity of the same img is ", float(sm.hist_intersection(img1, img2))
    print "Histogram Intersection similarity of 2 different samples is ", float(sm.hist_intersection(img1, img3))
