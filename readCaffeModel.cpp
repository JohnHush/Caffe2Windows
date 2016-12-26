#include "caffe.pb.h"
#include <opencv2/opencv.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "RedPixelDetector/mixed_gaussian_rpd.hpp"
#include "Boxdetector/line_box_detector.hpp"

using namespace std;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

#include <fcntl.h>
#include "Blob.hpp"
#include "RedPixelsExtractor.hpp"
#include "util.hpp"
#include "Binarizator/adaptive_threshold.hpp"

int main( void )
{
	const char * filename = "lenet_iter_10000.caffemodel";

	caffe::NetParameter net;

	fstream input( filename , ios::in | ios::binary);	
	net.ParseFromIstream( &input );
#ifndef DEBUG
	IplImage * imgSrc = cvLoadImage( "./test_data/line2_test.jpg" , CV_LOAD_IMAGE_COLOR );
	LineBoxDetector tst( imgSrc , 100 );
	tst.detectBox();
	tst.showOnImage();

	return 1;

	IplImage * imgThreshold = cvCreateImage( cvGetSize( imgSrc ) , 8 , 1 );

	AdaThre adapt_thresholder( 201 , 20 );
	adapt_thresholder.binarizate( imgSrc , imgThreshold );
	cvNamedWindow( "show" , CV_WINDOW_NORMAL );
	cvShowImage("show", imgThreshold);
	cvWaitKey();

	for ( int irow = 0 ; irow < imgSrc->height ; ++ irow )
	for ( int icol = 0 ; icol < imgSrc->width  ; ++ icol )
	{
		if ( cvGetReal2D( imgThreshold , irow , icol ) == 255 )
			cvSet2D( imgSrc , irow , icol , cvScalar(0,0,0) );
	}

	MixedGaussianRPD MGPRD( imgSrc );


	MGPRD.hasRedPixels();
	IplImage * imgcolor = cvCreateImage( cvSize( 28 , 28 ) , 8  , 1 );
	MGPRD.getRedPixels( imgcolor );

	cvNamedWindow( "show" , CV_WINDOW_NORMAL );
	cvShowImage("show", imgcolor );
	cvWaitKey();
	//test end

//	IplImage * imgRst = cvCreateImage( cvSize( 28 , 28 ) , 8 , 1 );
//	pair<float , float> MODEL_PRIOR = make_pair( 0.5 , 0.5 );
//	bool st = hasRedPixelsAndPickUp( imgSrc , imgRst , MODEL_PRIOR );

	vector<float> score;
	compute_score( imgcolor , net , score );

	for ( int i = 0 ; i < 10 ; i++)
		cout << "i = " << i << "  score = " << score[i] << endl;

	char s ;
	cin >>s;
#endif
// test on MNIST test set, the accuracy is 99.18%;
#ifdef DEBUG
	::std::ifstream image_file ( "t10k-images-idx3-ubyte" , std::ios::in | std::ios::binary );
    ::std::ifstream label_file ( "t10k-labels-idx1-ubyte" , std::ios::in | std::ios::binary );

    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
    uint32_t cols;

    image_file.read( reinterpret_cast<char *> (&magic) , 4 );
    image_file.read( reinterpret_cast<char *> (&num_items) , 4 );
    image_file.read( reinterpret_cast<char *> (&rows) , 4 );
    image_file.read( reinterpret_cast<char *> (&cols) , 4 );

    label_file.read( reinterpret_cast<char *> (&magic) , 4 );
    label_file.read( reinterpret_cast<char *> (&num_labels) , 4 );

    IplImage * imgSrc = cvCreateImage( cvSize( 28 , 28 ) , 8 , 1 );

    char * pixels = new char[28*28];
    char label;
	vector<float> score;
	int count = 0 ;

	cvNamedWindow("9", CV_WINDOW_NORMAL );
	
	for ( int i = 0 ; i < 10000 ; i ++ )
	{
		cout << "i = " << i << "  ";

    	image_file.read( pixels , 28* 28);
    	label_file.read( &label , 1 );

    	for ( int irow = 0 ; irow < 28 ; irow ++  )
	    {
			unsigned char * ptr = (unsigned char *)( imgSrc->imageData + irow * imgSrc->widthStep );
        	for ( int icol = 0 ; icol < 28 ; icol ++ )
			{
    	        ptr[icol] = pixels[ irow * 28 + icol ];
			}
    	}

		compute_score( imgSrc , net , score );
		if ( findMax( score ) == (int)( label + 0) )
			count ++;

		
		cvShowImage( "9" , imgSrc );
		cvWaitKey();
		

		if (findMax( score) == int( label + 0 ))
		{
			cout << "accuracy = " << 1.*count/(i+1) << endl;;
		}
		cout << " test of output = " << int( label ) << endl;
	}

#endif
//	delete coded_input;  
//	delete raw_input;  
//	close(fd);

	return 0;

}
