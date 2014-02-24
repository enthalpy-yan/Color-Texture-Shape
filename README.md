Color-Texture-Shape
===================

This repo is the Course Project 1 for CS598 Visual Information Retrieval in Stevens Institute of Technology which needs to implement the algorithms to extract the color, texture, and shape features including color histogram, color  correlogram, local binary pattern histogram, and histogram of oriented gradient. It also implements four different distance functions include L1, L2, Chi-Square, and the KL-distances, and two similarity functions include Cosine similarity and histogram intersection.

##Dependecies:

[OpenCV-Python](http://docs.opencv.org/trunk/doc/py_tutorials/py_setup/py_table_of_contents_setup/py_table_of_contents_setup.html#py-table-of-content-setup)

[Scipy Library](http://www.scipy.org/scipylib/index.html)

[Flask](http://flask.pocoo.org/) (back-end server in the final project)

[Polymer](http://www.polymer-project.org/) (front-end in the final project)

##File Structure:
```
Project1/
  |-modules/
  |--colorHistogram/          (Directory for source code that implements color histogram calculation)
  |-Descriptors.py            (Image descriptor functions)
  |-DistanceMeasures.py       (Distance Measurement functions)
  |-SimilarityMeasures.py     (Similarity Measurement functions)
  |-Main.py                   (Runnable file for testing our APIs)
```
