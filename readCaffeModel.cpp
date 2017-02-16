#include "caffe.pb.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "Boxdetector/line_box_detector.hpp"

using namespace std;
using namespace caffe;

#include <fcntl.h>
#include "util.hpp"
#include "Binarizator/adaptive_threshold.hpp"
#include "tools_classifier.hpp"

int main( int argc , char ** argv )
{
//	string name ( argv[1] );
	string file_name;

	IplImage * imgtst = cvLoadImage( "D:\\MyProjects\\orion-eye\\57.png" , CV_LOAD_IMAGE_COLOR );

	if ( imgtst == NULL )
		return -1;
	const char * filename = "D:\\MyProjects\\orion-eye\\lenet_iter_200.caffemodel";
	caffe::NetParameter net;
	fstream input( filename , ios::in | ios::binary);	
	net.ParseFromIstream( &input );

	AdaThre adapt_thresholder( 201 , 20 );

	IplImage * imgred = cvCreateImage( cvSize(28,28) , 8 , 1 );
	cvSetZero( imgred );

	bool hasma = jh::getRedPixelsInHSVRange( imgtst , adapt_thresholder , 0.1 , imgred );

	if ( !hasma )
	{
		cout << "the image is blank!\n";
		return -1;
	}

	vector<float> score;
	compute_score( imgred , net , score );

	for ( int i = 0 ; i < 10 ; i++)
		cout << "i = " << i << "  score = " << score[i] << endl;

	cout << "end of calculating the score!\n" << endl;
	char s;
	cin >>s;

	return 0;
}
