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
#define FINETUNE

int main( int argc , char ** argv )
{
	string file_name;

//	IplImage * imgtst = cvLoadImage( "/home/pitaloveu/orion-eye/test_data/color_5.jpg" , CV_LOAD_IMAGE_COLOR );
	IplImage * imgtst = cvLoadImage( argv[1] , CV_LOAD_IMAGE_COLOR );

	if ( imgtst == NULL )
    {
        printf( "we don't get an image!\n " );
		return -1;
    }
	const char * filename = "/home/pitaloveu/orion-eye/src/lenet_iter_10000.caffemodel";
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
#ifdef FINETUNE
	int input_label = 7;
//	IplImage * img_finetune = cvLoadImage( argv[1] , CV_LOAD_IMAGE_COLOR );
	string solver_prototxt( "/home/pitaloveu/orion-eye/src/lenet_solver_adam.prototxt" );
	string pretrained_model( "/home/pitaloveu/orion-eye/src/lenet_iter_10000.caffemodel" );
	finetune_model_by_caffe( solver_prototxt , pretrained_model , "" , imgred , input_label );
#endif

//	vector<float> score(10,0);
	string deployModel( "/home/pitaloveu/orion-eye/src/deploy_lenet.prototxt" );
	string caffeModel( "/home/pitaloveu/Desktop/retrained_iter_5.caffemodel" );
	vector<float> score = compute_score_by_caffe( imgred , deployModel , caffeModel );
//	compute_score( imgred , net , score );

	for ( int i = 0 ; i < 10 ; i++)
		cout << "i = " << i << "  score = " << score[i] << endl;

	cout << "end of calculating the score!\n" << endl;

	return 0;
}
