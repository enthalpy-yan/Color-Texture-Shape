#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

// for samples, and the variable "size" means the dimension of the sample vector
struct SampleVector 
{
	// the pointer points to a double type array, which contains each components of one sample vector,
	// for example, for a d-dimension sample vector X, coords[0], coords[1],..., coords[d-1] stand for its d components.
	double * coords; 	
	int size;//dimension
	SampleVector() :  coords(0), size(0) {} //constructor initializer list
	SampleVector(int d) { create(d); } // constructor overload 

	// to construct a d-dimension vector with the initialized components of zero's.
	void create(int d)
	{
		size = d;
		coords = new double[size];//dynamically create a doulbe-type array, the number of elements is "size".
		for (int i=0; i<size; i++)//to initialize each component as zero
			coords[i] = 0.0;
	}

	// to copy a SampleVector-type variable
	void copy(const SampleVector & other)
	{
		if (size == 0) // if no data exists, then a new one will be created.
			create(other.size);

		for (int i=0; i<size; i++)
			coords[i] = other.coords[i];
	}

	// to add two SampleVector-type variables, like X and Y,
	// we can get X = X + Y, i.e., X = [X1 + Y1, X2+Y2, ... ,X(size-1)+ Y(size-1)].
	void add(const SampleVector& other)
	{
		for (int i=0; i<size; i++)
			coords[i] += other.coords[i];
	}
	// deconstructor
	~SampleVector()
	{
		if(coords)// if "coords" is non-zero.
			delete[] coords;
		size = 0;
	}
};


// Cluster type variable, defined for the classified results of K-Means algorithm. 
struct Cluster 
{
	SampleVector center;    // the center of the cluster
	int*   member;    // the save the indices of the samples in the cluster
	int    memberNum; // the number of samples classified to the cluster
};

// the class of KMeans
class KMeans
{
private:
	int      step;//to draw a color histogram, with the bin's width being "step" pixels, 	
	int      hist_height;// and with the maximun height being "hist_height" pixels.
	int      num;          // the number of samples to be classified;
	int      dimen;        // the dimension of samples to be classified;
	int      clusterNum;   // the number of the clusters for K-Means algorithm
	SampleVector*  observations; // points to the array containing all of the samples 
	Cluster* clusters;     // for each cluster
	int      passNum;      // iteration times

public:
	// constructor
	KMeans(int stepp, int hist_heightt,int n, int d, int k, SampleVector* ob);

	// deconstructor
	~KMeans();

    // for the first iteration of the K-Means algorithm,
	// we select the first k samples as the centers of the k clusters, respectively
	void initClusters();

	// the run function can achieve the implementing of the K-Means algorithm,
	// through calling some major functions, 
	// including distribute(), recalculateCenters(),showK_MeansResults() and drawHistogram();
	void run();

	// to print the results of the K-Means algorithm.
	void showK_MeansResults();

	void distribute();//for each sample, to classify it to its corresponding cluster, and save its index

	// to return the closest cluster ID for the id-th sample.
	int closestCluster(int id);

	// to get the euclidean distance from the id-th sample to the k-th cluster.
	double eucNorm(int id, int k);

	//re-calculate the new centers of k clusters, respectively
    //until the centers will not change.
	bool recalculateCenters();

	//to return the cluster with the "id" index
	Cluster& getCluster(int id);

	// to draw a color histogram based on the classification results of K-Means algorithm.
	void KMeans::drawHistogram();

};// the end of Class KMeans definition KMeans


// to print a "size" dimension sample vector
void printVector(ostream& output, const SampleVector& v);

// to get the samples to be classified from the input color image.
SampleVector * getSamplesFromColorImage(Mat & image,// color image to be loaded
	int smaplesDimen,int clustersNum// Clusters Number is difined by users;
	); 