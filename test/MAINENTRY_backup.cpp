#include "caffe/proto/caffe.pb.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "Boxdetector/line_box_detector.hpp"
#include "caffe/caffe.hpp"

using namespace std;
using namespace caffe;

#include <fcntl.h>
#include "util.hpp"
#include "Binarizator/adaptive_threshold.hpp"
#include "tools_classifier.hpp"
//#define FINETUNE

int main( int argc , char ** argv )
{
	string file_name;
#ifdef UNIX

	IplImage * imgtst = cvLoadImage( argv[1] , CV_LOAD_IMAGE_COLOR );
#endif
#ifdef _WINDOWS
	IplImage * imgtst = cvLoadImage( "D:\\MyProjects\\orion-eye\\test_data\\color_5.jpg" , CV_LOAD_IMAGE_COLOR );
#endif

	if ( imgtst == NULL )
    {
        printf( "we don't get an image!\n " );
		return -1;
    }
    
	AdaThre adapt_thresholder( 201 , 20 );

	IplImage * imgred = cvCreateImage( cvSize(28,28) , 8 , 1 );
	cvSetZero( imgred );

	bool hasma = jh::getRedPixelsInHSVRange( imgtst , adapt_thresholder , 0.05 , imgred );
#ifdef DEBUG
	showImage( imgred , 10 , "red" );
#endif

	if ( !hasma )
	{
		cout << "the image is blank!\n";
		return -1;
	}
#ifdef FINETUNE
	int input_label = argv[2][0] - '0';
	finetune_by_caffe( "lenet_FINETUNE.caffemodel" , "lenet_train.prototxt" , imgred , input_label );
#endif
#ifdef UNIX
	string deployModel( "/home/pitaloveu/orion-eye/build/src_build/deploy_lenet.prototxt" );
	string caffeModel( "/home/pitaloveu/orion-eye/build/src_build/lenet_FINETUNE.caffemodel" );
#endif
#ifdef _WINDOWS
	string deployModel( "D:\\MyProjects\\orion-eye\\deploy_lenet.prototxt" );
	string caffeModel( "D:\\MyProjects\\orion-eye\\lenet_FINETUNE.caffemodel" );
#endif
	vector<float> score = compute_score_by_caffe( imgred , deployModel , caffeModel );

	for ( int i = 0 ; i < 10 ; i++)
		cout << "i = " << i << "  score = " << score[i] << endl;

	char sb;
	std::cin >> sb;
	return 0;
}
