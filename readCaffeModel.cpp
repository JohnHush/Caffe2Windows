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
//	IplImage* src = cvLoadImage( "./test_data/line2_test.jpg" , CV_LOAD_IMAGE_COLOR );
//	IplImage* dst = cvCreateImage( cvGetSize(src), 8, 1 );
//	IplImage* color_dst = cvCreateImage( cvGetSize(src), 8, 3 );
//	CvMemStorage* storage = cvCreateMemStorage(0);//存储检测到线段,当然可所以N*1的矩阵数列,若是
//	CvSeq* lines = 0;
//	int i;
//
//	IplImage* src1=cvCreateImage(cvSize(src->width,src->height),IPL_DEPTH_8U,1);
//
//	cvCvtColor(src, src1, CV_BGR2GRAY); //把src转换成灰度图像保存在src1中,重视进行边沿检测必然要
//
//	cvCanny( src1, dst, 50, 200, 3 );//参数50,200的灰度变换
//
//	cvCvtColor( dst, color_dst, CV_GRAY2BGR );
//#if 0
//	lines = cvHoughLines2( dst, storage, CV_HOUGH_STANDARD, 1, CV_PI/180, 100, 0, 0 );//标准霍夫变换后两个参数为0,因为line_storage是内存空间,所以返回一个CvSeq序列布局的指针
//
//	for( i = 0; i < lines->total; i++ )
//	{
//		float* line = (float*)cvGetSeqElem(lines,i);//用GetSeqElem获得直线
//		float rho = line[0];
//		float theta = line[1];//对于SHT和MSHT(标准变换)这里line[0],line[1]是rho(与像素相干单位的距离精度)和theta(弧度测量的角度精度)
//		CvPoint pt1, pt2;
//		double a = cos(theta), b = sin(theta);
//		if( fabs(a) < 0.001 )
//		{
//			pt1.x = pt2.x = cvRound(rho);
//			pt1.y = 0;
//			pt2.y = color_dst->height;
//		}
//		else if( fabs(b) < 0.001 )
//		{
//			pt1.y = pt2.y = cvRound(rho);
//			pt1.x = 0;
//			pt2.x = color_dst->width;
//		}
//		else
//		{
//			pt1.x = 0;
//			pt1.y = cvRound(rho/b);
//			pt2.x = cvRound(rho/a);
//			pt2.y = 0;
//		}
//		cvLine( color_dst, pt1, pt2, CV_RGB(255,0,0), 3, 8 );
//	}
//#else
//	lines = cvHoughLines2( dst, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI/180, 50, 30, 5 );
//	for( i = 0; i < lines->total; i++ )
//	{
//		CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
//		cvLine( color_dst, line[0], line[1], CV_RGB(255,0,0), 3, 8 );
//	}
//#endif
//
//	cvNamedWindow( "Source", CV_WINDOW_NORMAL );
//	cvShowImage( "Source", src );
//
//	cvNamedWindow( "Hough", CV_WINDOW_NORMAL );
//	cvShowImage( "Hough", color_dst );
//
//	cvWaitKey(0);

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
