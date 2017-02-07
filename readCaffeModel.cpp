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

	IplImage * imgSrc1 = cvLoadImage( "./test_data/TEST_SET/y-4-3.jpg" , CV_LOAD_IMAGE_COLOR );
	IplImage * imgSrc2 = cvLoadImage( "./test_data/TEST_SET/g-2-1.jpg" , CV_LOAD_IMAGE_COLOR );
	IplImage * imgSrc3 = cvLoadImage( "./test_data/TEST_SET/y-6-3-1.jpg" , CV_LOAD_IMAGE_COLOR );

	IplImage * imgSrc = cvLoadImage( "./test_data/TEST_SET/y-6-6.jpg" , CV_LOAD_IMAGE_COLOR );

	if ( imgSrc == NULL )
		return 1;

	AdaThre adapt_thresholder( 201 , 20 );

	IplImage * imgcolor = cvCreateImage( cvSize( 28 , 28 ) , 8  , 1 );

	vector<IplImage *> imgs(3);
	imgs[0] = imgSrc1;
	imgs[1] = imgSrc2;
	imgs[2] = imgSrc3;

	jh::mg_classifier mgc;

	jh::train_classifier( imgs , adapt_thresholder , 20 , 200 ,  mgc );

	bool flag = jh::getRedPixels( imgSrc , adapt_thresholder , mgc , 20 , 0.01 , 0.8, imgcolor);

	if ( flag == false )
	{
		cout << "the image is blank " << endl;
		return 1;
	}

	cout << "start to calculating the score!\n" << endl;

	vector<float> score;
	compute_score( imgcolor , net , score );

	for ( int i = 0 ; i < 10 ; i++)
		cout << "i = " << i << "  score = " << score[i] << endl;

	cout << "end of calculating the score!\n" << endl;

	return 0;
}
