#include "HandWritingDigitsRecognitionSystem.h"

#include "caffe.pb.h"
#include <opencv2/opencv.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <time.h>
#include <windows.h>

using namespace std;

#include <fcntl.h>
#include "util.hpp"
#include "Binarizator/adaptive_threshold.hpp"
#include "Classifier/Mixed_Gaussian_Classifier.hpp"
#include "tools_classifier.hpp"

AdaThre * adapt_thresholder = nullptr;
jh::mg_classifier * mgc = nullptr;

float * imgData;
float * imgData_col;
float * conv1_col;
float * pool1_col;
float * pool1_2_mat;
float * conv2_col;
float * pool2_col;
float * inner_r1;
float * inner_r2;

float * k1;
float * b1;
float * k2;
float * b2;
float * inner_w1;
float * inner_w2;
float * inner_b1;
float * inner_b2;

int epsilon_;
int iteration_;

void initPredictor( int BLOCK_SIZE , double OFFSET , int epsilon, int iteration , vector<IplImage *> & imgs )
{
	epsilon_ = epsilon;
	iteration_ = iteration;
	caffe::NetParameter net;
	fstream input( "lenet_iter_10000.caffemodel" , ios::in | ios::binary);
	net.ParseFromIstream( &input );

	adapt_thresholder = new AdaThre( BLOCK_SIZE , OFFSET );
	mgc = new jh::mg_classifier;

	if ( imgs.size() == 0 )
	{
		imgs.resize(1);
		IplImage * imgSrc1 = cvLoadImage( "g-2-1.jpg" , CV_LOAD_IMAGE_COLOR );
//		IplImage * imgSrc2 = cvLoadImage( "g-illegal-1.jpg" , CV_LOAD_IMAGE_COLOR );
//		IplImage * imgSrc3 = cvLoadImage( "y-4-3.jpg" , CV_LOAD_IMAGE_COLOR );
//		IplImage * imgSrc4 = cvLoadImage( "y-6-3-1.jpg" , CV_LOAD_IMAGE_COLOR );

		imgs[0] = imgSrc1;
//		imgs[1] = imgSrc2;
//		imgs[2] = imgSrc3;
//		imgs[3] = imgSrc4;
	}

	jh::train_classifier( imgs , *adapt_thresholder , epsilon , iteration ,  *mgc );

	imgData		= new float [ 28 * 28 ];
	k1			= new float [ 20 * 25 ];
	b1			= new float [ 20 ];
	imgData_col = new float [ 24 * 24 * 25 ];
	conv1_col	= new float [ 20 * 24 * 24 ];
	pool1_col	= new float [ 20 * 12 * 12 ];

	pool1_2_mat	= new float [ 8 * 8 * 20 * 5 * 5 ];
	k2			= new float [ 50 * 20 * 5 * 5 ];
	b2			= new float [ 50 ];
	conv2_col	= new float [ 50 * 8 * 8 ];
	pool2_col	= new float [ 50 * 4 * 4 ];

	inner_w1	= new float [ 500 * 800 ];
	inner_r1	= new float [ 500 ];
	inner_w2	= new float [ 10 * 500 ];
	inner_r2	= new float [ 10 ];

	inner_w1	= new float [ 500 * 800 ];
	inner_b1	= new float [ 500 ];
	inner_r1	= new float [ 500 ];
	inner_w2	= new float [ 10 * 500 ];
	inner_b2	= new float [ 10 ];
	inner_r2	= new float [ 10 ];

	for ( int i = 0 ; i < 20 * 25 ; ++i )
		k1[i] = net.layer(1).blobs(0).data(i);

	for ( int i = 0 ; i < 20 ; ++i )
		b1[i] =  net.layer(1).blobs(1).data(i);

	for ( int i = 0 ; i < 50 * 20 * 5 * 5 ; ++i )
		k2[i] = net.layer(3).blobs(0).data(i);
	for ( int i = 0 ; i < 50 ; i ++ )
		b2[i] = net.layer(3).blobs(1).data(i);

	for ( int i = 0 ; i < 500 * 800 ; i++ )
		inner_w1[i] = net.layer(5).blobs(0).data(i);
	for ( int i = 0 ; i < 500 ; i ++ )
		inner_b1[i] = net.layer(5).blobs(1).data(i);

	for ( int i = 0 ; i < 10 * 500 ; i ++ )
		inner_w2[i] = net.layer(7).blobs(0).data(i);
	for ( int i = 0 ; i < 10 ; i ++ )
		inner_b2[i] = net.layer(7).blobs(1).data(i);
}

void deletePredictor()
{
	delete adapt_thresholder;
	delete [] imgData;
	delete [] k1;
	delete [] b1;
	delete [] imgData_col;
	delete [] conv1_col;
	delete [] pool1_col;
	delete [] pool1_2_mat;
	delete [] k2;
	delete [] b2;
	delete [] conv2_col;
	delete [] pool2_col;
	delete [] inner_w1;
	delete [] inner_r1;
	delete [] inner_w2;
	delete [] inner_r2;
	delete [] inner_b1;
	delete [] inner_b2;
}

int looksLikeNumber( IplImage * imgSrc , float if_less_than_then_its_blank  , float keep_at_least_area  )
{
	//std::cout << "start doing adaptive thresholding \n" << std::endl;
	//clock_t start_threshold = clock();

	//IplImage * imgThreshold = cvCreateImage( cvGetSize( imgSrc ) , 8 , 1 );
	//adapt_thresholder->binarizate( imgSrc , imgThreshold );
	//for ( int irow = 0 ; irow < imgSrc->height ; ++ irow )
	//for ( int icol = 0 ; icol < imgSrc->width  ; ++ icol )
	//{
	//	if ( cvGetReal2D( imgThreshold , irow , icol ) == 255 )
	//		cvSet2D( imgSrc , irow , icol , cvScalar(0,0,0) );
	//}

	////clock_t end_threshold = clock();
	////std::cout << "time of adaptive thresholding is = " << (end_threshold - start_threshold) << endl;
	////std::cout << "start doing Mixed Gaussian separation \n" << std::endl;
	////clock_t start_MGRPD = clock();

	//MixedGaussianRPD MGPRD( imgSrc );
	//MGPRD.hasRedPixels();
	//if ( !MGPRD.redOrNot())
	//{
	//	cvReleaseImage( &imgThreshold );
	//	return -1;
	//}

	//clock_t end_MGRPD = clock();
	//std::cout << "time of Mixed Gaussian is = " << (end_MGRPD - start_MGRPD) << endl;
	//std::cout << "start doing getting red pixels \n" << std::endl;
	//clock_t start_getred = clock();
	clock_t start_net = clock();

	IplImage * imgcolor = cvCreateImage( cvSize( 28 , 28 ) , 8  , 1 );

	bool flag = jh::getRedPixels( imgSrc , *adapt_thresholder , *mgc , epsilon_ , if_less_than_then_its_blank , keep_at_least_area , imgcolor);

	if ( flag == false )
		return -1;

//	MGPRD.getRedPixels( imgcolor );

	//clock_t end_getred = clock();
	//std::cout << "time of Getting Red is = " << (end_getred - start_getred) << endl;
	//std::cout << "start doing prediction using the CNN_NET \n" << std::endl;
	

	vector<float> score(10);
	for ( int irow = 0 ; irow < 28 ; ++ irow )
	for ( int icol = 0 ; icol < 28 ; ++ icol )
	{
		imgData[ irow * 28 + icol ] = cvGetReal2D( imgcolor , irow , icol ) * 0.00390625;
	}
	im2col( imgData , 1 , 28, 28 , 5 , 5 , imgData_col );
	wrapper_cblas_gemm<float>( CblasNoTrans , CblasNoTrans , 20 ,  24*24 , 25 , 1., k1 ,imgData_col , 0. , conv1_col );
	for ( int i = 0 ; i < 20 * 24 * 24 ; ++ i )
	{
		conv1_col[i] += b1[i/(24*24)];
	}
	pooling( conv1_col , pool1_col , 20 , 24 , 24 );
	im2col( pool1_col , 20 , 12 , 12 , 5 , 5 , pool1_2_mat );
	wrapper_cblas_gemm<float>( CblasNoTrans , CblasNoTrans , 50, 8*8 , 20*5*5, 1., k2 ,pool1_2_mat, 0. , conv2_col );
	for ( int i =0 ; i < 50 * 8 * 8 ; i ++ )
		conv2_col[i] += b2[i/(8*8)];
	pooling( conv2_col , pool2_col , 50 , 8 , 8 );
	wrapper_cblas_gemv<float>( CblasNoTrans , 500 ,800,1.,inner_w1, pool2_col,0,inner_r1);
	for ( int i = 0 ; i < 500 ; i ++ )
		inner_r1[i] += inner_b1[i];
	relu( inner_r1 , 500 );
	wrapper_cblas_gemv<float>( CblasNoTrans ,10 ,500,1.,inner_w2, inner_r1,0,inner_r2 );
	for ( int i = 0 ; i < 10 ; i ++ )
		inner_r2[i] += inner_b2[i];
	for ( int i = 0 ; i < 10 ; i++ )
		score[i] = inner_r2[i];

	clock_t end_net = clock();
	std::cout << "time of IDENTIFY is = " << (end_net - start_net) << endl;

	//cvReleaseImage( &imgThreshold );
	cvReleaseImage( &imgcolor );

	return findMax( score );

}
