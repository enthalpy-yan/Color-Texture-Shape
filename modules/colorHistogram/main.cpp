#include  "KMeans.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(){

    // read an image
	cv::Mat image= cv::imread("D:\\baboon.jpg");	
	// create image window named "My Image"
	cv::namedWindow("My Image");
	// show the image on window
	cv::imshow("My Image", image);
	// wait key for 5000 ms
	cv::waitKey(5000);
	
	int samplesNum = (image.cols)*(image.rows);
	SampleVector * obs = getSamplesFromColorImage(image,3,10);
	KMeans kmeans(20,256,samplesNum,3,30,obs);
	delete [] obs;
	return 0;
}
