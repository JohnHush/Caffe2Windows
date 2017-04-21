#define NOMINMAX
#define NO_STRICT
#include "HandWritingDigitsRecognitionSystem.h"

#ifdef BUILD_OCR_PREDICT
#include "caffe.pb.h"
#else
#include <caffe/proto/caffe.pb.h>
#endif

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <time.h>
#include "config.hpp"
#include <fcntl.h>
#include "util.hpp"
#include "Binarizator/adaptive_threshold.hpp"
#include "Classifier/Mixed_Gaussian_Classifier.hpp"
#include "tools_classifier.hpp"

using namespace std;
AdaThre * adapt_thresholder = NULL;

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

void initPredictor( int BLOCK_SIZE , double OFFSET  )
{
#ifdef _WINDOWS
	CHAR exeFullPath[MAX_PATH];
	string strPath;
	GetModuleFileNameA(NULL,exeFullPath,MAX_PATH);
	strPath = exeFullPath;
	strPath = strPath.substr( 0 , strPath.rfind('\\') +1 );
#endif
#ifdef UNIX
	int MAXBUFSIZE = 1024;
    int count;
    char buf[MAXBUFSIZE];

    count = readlink( "/proc/self/exe" , buf , MAXBUFSIZE );
//    if ( count < 0 || count >= MAXBUFSIZE )
//        LOG(FATAL) << "size of the exe path wrong !" << std::endl;

    string strPath( buf );
    strPath = strPath.substr( 0 , strPath.rfind( '/' ) + 1 );
    //LOG(INFO) << strPath << std::endl;
#endif

#ifdef APPLE
    char exeFullPath[1024];
    unsigned size = 1024;
    string strPath;
    _NSGetExecutablePath( exeFullPath , &size );
    exeFullPath[size] = '\0';
    strPath = exeFullPath;
    strPath = strPath.substr( 0 , strPath.rfind( '/' )+1 );
#endif

	string modStr = strPath + "lenet_FINETUNE.caffemodel";
	caffe::NetParameter net;
	fstream input( modStr , ios::in | ios::binary);
	net.ParseFromIstream( &input );

	adapt_thresholder = new AdaThre( BLOCK_SIZE , OFFSET );

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

 int looksLikeNumber( IplImage * imgSrc   , IplImage* imgOut , float & confidence , float red_pts_prec  )
{
	IplImage * imgcolor = cvCreateImage( cvSize( 28 , 28 ) , 8  , 1 );
	
	cvSetZero( imgcolor );

	bool hasma = jh::getRedPixelsInHSVRange2( imgSrc , *adapt_thresholder , red_pts_prec , imgcolor );	

//	cvReleaseImage( &imgSrc );
//	imgOut = cvCreateImage( cvSize(280 , 280) , 8 , 1 );
	cvSetZero( imgOut );

	if ( !hasma ) 
	{
		cvReleaseImage( &imgcolor );
		confidence = 1;
		return -1 ;
	}

	for ( int irow = 0 ; irow < 280 ; ++ irow )
	for ( int icol = 0 ; icol < 280 ; ++ icol )
	{ 
		cvSetReal2D( imgOut , irow , icol , cvGetReal2D( imgcolor , irow/10 , icol/10 ) );
	}

//	string te2("1111111end");
//	MessageBoxA(NULL,"1111111end","1111111end",MB_OK|MB_SYSTEMMODAL);

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

	// add softmax loss layer
	
	int max_index = findMax( score );
	for ( int i = 0 ; i < 10 ; i ++ )
		inner_r2[i] = score[i] - score[max_index];

	float sum_exp = 0.;
	for ( int i = 0 ; i < 10 ; i ++ )
		sum_exp += std::exp( inner_r2[i] );

	for ( int i = 0 ; i < 10 ; i ++ )
		score[i] = std::exp( inner_r2[i] ) / sum_exp;

	confidence = score[ max_index ];

	cvReleaseImage( &imgcolor );

	return max_index ;

}
