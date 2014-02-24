Color-Texture-Shape
===================

This repo is the Course Project 1 for CS598 Visual Information Retrieval in Stevens Institute of Technology which implements the algorithms to extract the color, texture, and shape features including color histogram, color  correlogram, local binary pattern histogram, and histogram of oriented gradient. It also implements four different distance functions include L1, L2, Chi-Square, and the KL-distances, and two similarity functions include Cosine similarity and histogram intersection.

##Dependecies:

[OpenCV-Python](http://docs.opencv.org/trunk/doc/py_tutorials/py_setup/py_table_of_contents_setup/py_table_of_contents_setup.html#py-table-of-content-setup)

[Scipy Library](http://www.scipy.org/scipylib/index.html)

[Numpy] (http://www.numpy.org)

##File Structure:
```
root/
  |-lib/                      (Directory for holding library files)
  |--Descriptors.py            (Image descriptor functions)
  |--DistanceMeasures.py       (Distance Measurement functions)
  |--SimilarityMeasures.py     (Similarity Measurement functions)
  |-modules/                  (Directory for holding modules implemented using c++)
  |--colorHistogram/          (Source code that implements color histogram calculation)
  |-Main.py                   (Runnable file for testing our APIs)
```
## How to run the code:
1. Install dependecies (OpenCV-Python, Scipy, Numpy)
2. Open a command line tool
3. python Main.py
