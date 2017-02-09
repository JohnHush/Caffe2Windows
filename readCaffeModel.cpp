#include "caffe.pb.h"
#include <opencv2/opencv.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
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
#include "util.hpp"
#include "Binarizator/adaptive_threshold.hpp"
#include "tools_classifier.hpp"
//#include "HandWritingDigitsRecognitionSystem.h"

int main( void )
{
	const char * filename = "lenet_iter_10000.caffemodel";
	caffe::NetParameter net;
	fstream input( filename , ios::in | ios::binary);	
	net.ParseFromIstream( &input );

	AdaThre adapt_thresholder( 201 , 20 );

	IplImage * imgtst = cvLoadImage( "/home/pitaloveu/Desktop/2.png" , CV_LOAD_IMAGE_COLOR );
	bool hasma = jh::hasPixelsInBox( imgtst , adapt_thresholder , 20 , 0.01 );

//	IplImage * imgSrc1 = cvLoadImage( "./test_data/TEST_SET/y-4-3.jpg" , CV_LOAD_IMAGE_COLOR );
//	IplImage * imgSrc2 = cvLoadImage( "./test_data/TEST_SET/g-2-1.jpg" , CV_LOAD_IMAGE_COLOR );
//	IplImage * imgSrc3 = cvLoadImage( "./test_data/TEST_SET/y-6-3-1.jpg" , CV_LOAD_IMAGE_COLOR );

//	IplImage * imgSrc = cvLoadImage( "./test_data/TEST_SET/q-4-3-1.jpg" , CV_LOAD_IMAGE_COLOR );

//	if ( imgSrc == NULL )
//		return 1;

	if ( !hasma )
		return -1;

	IplImage * imgcolor = cvCreateImage( cvSize( 28 , 28 ) , 8  , 1 );

	vector<IplImage *> imgs(1);
	imgs[0] = imgtst;

	jh::mg_classifier mgc;

	jh::train_classifier( imgs , adapt_thresholder , 20 , 200 ,  mgc );

	bool flag = jh::getRedPixels( imgtst , adapt_thresholder , mgc , 20 , 0.01 , 0.8, imgcolor);

	if ( flag == false )
	{
		cout << "the image is blank " << endl;
		return -1;
	}

	cout << "start to calculating the score!\n" << endl;

	vector<float> score;
	compute_score( imgcolor , net , score );

	for ( int i = 0 ; i < 10 ; i++)
		cout << "i = " << i << "  score = " << score[i] << endl;

	cout << "end of calculating the score!\n" << endl;
	char s;
	cin >>s;

	return 0;
}
