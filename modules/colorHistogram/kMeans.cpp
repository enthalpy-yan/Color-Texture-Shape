#include "KMeans.h"

// KMeans class definition


	

// constructor
KMeans::KMeans(int stepp, int hist_heightt,int n, int d, int k, SampleVector* ob)
	: step(stepp) //to draw a color histogram, with the bin's width being "step" pixels,
	, hist_height(hist_heightt) //to draw a color histogram, with the maximun height being "hist_height" pixels.
    ,num(n) // the number of samples to be classified;
	, dimen(d) // the dimension of samples to be classified;
	, clusterNum(k) // the number of the clusters for K-Means algorithm
	, observations(ob) // points to the array containing all of the samples
	, clusters(new Cluster[k]) // for each cluster
{
	for (int k=0; k<clusterNum; k++)
		clusters[k].member = new int[n];
	// here the array of "member" contains "n" elements instead of "Cluster.memberNum";
	// usually Cluster.memberNum < n, but when the cluster number is only one, there will be an extreme case of "Cluster.memberNum = n".

	initClusters();
	run();
}


// deconstructor
KMeans::~KMeans()
{
	for (int k=0; k<clusterNum; k++)
		delete [] clusters[k].member;
	delete [] clusters;
}

void KMeans::initClusters()
{
	// for the first iteration of the K-Means algorithm,
	// we select the first k samples as the centers of the k clusters, respectively
	for (int i=0; i<clusterNum; i++)
	{
		clusters[i].member[0] = i;  // save the index of the sample vector classified into the i-th cluster
		clusters[i].center.copy(observations[i]); // and let this sample be the center for the first iteration.
	}
}

void KMeans::run()
{
	bool converged = false; // converged or not
	passNum = 0;//to identify iteration times
	while (!converged && passNum < 999)   // if not converged, then continue the iteration
		// usually the iteration will be finally converged, but let "passNum < 999" for avoiding abnormal cases.
	{
		distribute(); // for each sample, to classify it to its corresponding cluster, and save its index.
		converged = recalculateCenters(); // re-calculate the new centers of k clusters, respectively, until the centers will not change.
		passNum++;
	}

	showK_MeansResults();// to print the results of the K-Means algorithm.
    drawHistogram();// to draw a color histogram based on the classification results of K-Means algorithm.
}

void KMeans::distribute()
{
	// clean all the indices of the samples classified into the k-th cluster for the last iteration,
	// and do the re-allocation for the current iteration
	for(int k=0; k<clusterNum; k++)
		getCluster(k).memberNum = 0;

	// for each sample, to classify it to its corresponding cluster, and save its index
	// that is, if there are "num" samples, the i-th (i = 1, 2, 3, ..., num)sample classified to the k-th cluster,
	// will have the index of "i" in the k-th cluster.
	for(int i=0; i<num; i++)
	{
		// for the i-th sample, to find its corresponding cluster in terms of the shortest distance.
		Cluster& cluster = getCluster(closestCluster(i)); 
		
		//"memberNum" means the number of the samples in the k-th cluster,
		// and also means the position of the newly classified sample in the array of "member".
		cluster.member[cluster.memberNum] = i;  // save the index of the newly classified sample
		(cluster.memberNum)++;  //automatically increase the munber of the samples in the k-th cluster for the next iteration.
	}
}

// to return the closest cluster ID for the id-th sample.
int KMeans::closestCluster(int id)
{
	int clusterID = 0;  // firstly, we assume that the first cluster is closest to the id-th sample.
	double minDist = eucNorm(id, 0); // to calculate the euclidean distance from the id-th sample to the 0-th cluster
	
	// to calculate the euclidean distance from the id-th sample to the remaining clusters, i.e. do "Sorting".
	for (int k=1; k<clusterNum; k++) 
	{
		double d = eucNorm(id, k);
		if(d < minDist)// do "Sorting";
		{
			minDist = d;
			clusterID = k;
		}
	}
	return clusterID;
}

// to get the euclidean distance from the id-th sample to the k-th cluster.
double KMeans::eucNorm(int id, int k)
{
	SampleVector& observ = observations[id];
	SampleVector& center = clusters[k].center;
	double sumOfSquare = 0;

	// to sum the euclidean magnitude for a d-dimension vector
	for (int d=0; d<dimen; d++)
	{
		double dist = observ.coords[d] - center.coords[d];// for the d-th dimension
		sumOfSquare += dist*dist;
	}
	return sumOfSquare;
}


// re-calculate the new centers of k clusters, respectively
// until the centers will not change
bool KMeans::recalculateCenters()
{
	bool converged = true;

	for (int k=0; k<clusterNum; k++)
	{
		Cluster& cluster = getCluster(k);

		// to build a "dimen" dimension vector - average, as the initial center of the k-th cluster,
		// wherein k = 0, 1, 2, ..., clusterNum.
		SampleVector average(dimen);

		// to sum all the sample vectors in the k-th cluster, since the constructor of the class KMeans
		// will initialize all the sample components as zero's.
		for (int m=0; m<cluster.memberNum; m++)
			average.add(observations[cluster.member[m]]);

		//to sum all the dimension components for the summation of all the sample vectors in k-th cluster and then average it.
		for(int d=0; d<dimen; d++)
		{
			average.coords[d] /= cluster.memberNum;
           // if the current center is not the same as the previous center, please keep doing the iteration
			if(average.coords[d] != cluster.center.coords[d])
			{
				converged = false;
				cluster.center.coords[d] = average.coords[d]; // to get a new center for the k-th cluster
			}
		}
	}
	return converged;
}

// to print the results of the K-Means algorithm.
void KMeans::showK_MeansResults(){
	ostream& output = std::cout;
	for (int c=0; c<clusterNum; c++)
	{
		Cluster& cluster = this->getCluster(c);
		output << "---- For the " << (c + 1) << "-th cluster ----\n"; // display the c-th cluster
		output << "cluster center£¨"<<"dimension: "<<cluster.center.size<<"): ";

		printVector(output, cluster.center);
/*
		for (int m=0; m<clusters[c].memberNum; m++)
		{
			int id = cluster.member[m];
			printVector(output, observations[id]);
			cout<<endl;
		}
		output << endl;
*/

		std::cout<<endl;}
}

//to return the cluster with the "id" index
Cluster& KMeans::getCluster(int id)
{
	return clusters[id];
}

// to draw the color histogram
void KMeans::drawHistogram(){

	cv::Mat hist_img = cv::Mat::zeros(hist_height,clusterNum*step,CV_8UC3);
	
	double max_val= 0.0;// find the maximum number of samples in each cluster for normalization
	for(int j =0; j< clusterNum; j++ ){
			if (max_val <= clusters[j].memberNum)
				max_val = clusters[j].memberNum;
			if (j == 0)
				cout<<endl<<"the number of samples in each cluster is:"<<endl<<clusters[j].memberNum;
			else 
				cout<<","<<clusters[j].memberNum;
		}
		cout<<endl;
		cout<<endl;

	for (int i = 0; i< clusterNum; i++ ){
		int bin_height= int(clusters[i].memberNum*hist_height/max_val);
		cout<<i+1<<" bin's height is "<<bin_height<<endl;
		
		// Scalar( a, b, c ),We would be defining a RGB color such as: Red = c, Green = b and Blue = a
		rectangle(hist_img,cv::Point(i*step,hist_height-1),Point((i+1)*step-1,hist_height-1-bin_height),
			Scalar(clusters[i].center.coords[0],//For Blue Component
			clusters[i].center.coords[1],// For Green Component
			clusters[i].center.coords[2]),// For Red Component
			CV_FILLED
		);
	}	
	cv::namedWindow("ColorHistogram");
	// show the image on window
	cv::imshow("ColorHistogram", hist_img);
	// wait key for 5000 ms
	cv::waitKey(0);
		}
// the end of Class KMeans definition


// to print a "size" dimension sample vector
void printVector(ostream& output, const SampleVector& v)
{
	for (int i=0; i<v.size; i++)
	{
		if(i != 0)
			output << ",";
		output << v.coords[i];
	}
}


// get the samples to be classified from the input color image.
SampleVector * getSamplesFromColorImage(Mat & image,// color image to be loaded
	int smaplesDimen= 3,int clustersNum= 10// Clusters Number is difined by users;
	){
	int samplesNum = (image.cols)*(image.rows);
	SampleVector * obs = new SampleVector[int(samplesNum)];
	static int obs_id = 0; 
	if (image.channels() == 3){ // color image
		
		for (int j = 0; j < image.rows; j++){
			for (int i = 0; i < image.cols; i++){ 

			//This is an important sentence to create the data
			//if it were not written, the error of memory address would come.
              obs[obs_id].create(smaplesDimen); 

			  obs[obs_id].coords[0] = double(image.at<Vec3b>(j,i)[0]); // blue
			  obs[obs_id].coords[1] = double(image.at<Vec3b>(j,i)[1]); // green
			  obs[obs_id].coords[2] = double(image.at<Vec3b>(j,i)[2]); // red
			  obs_id ++;
			}
		}
	}
	
	else 
	std::cout<<" The input image is not a color one, please load a color image!"<<endl;

	std::cout<<"The number of all the samples in the image to be classified is "<<obs_id<<endl;

	return obs;
}

	