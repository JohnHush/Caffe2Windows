#include "util.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
#include <fstream>

using std::cout;
using std::endl;


void showImageMat(const cv::Mat & imgSrc, const float ratio, const string & windowName, const int waitingTime)
{
//	int width = int(imgSrc->width  * ratio);
//	int heigh = int(imgSrc->height * ratio);

//	IplImage * imgRst = cvCreateImage(cvSize(width, heigh), imgSrc->depth, imgSrc->nChannels);
//	cvResize(imgSrc, imgRst);

	cv::Mat imgRst;
	cv::resize( imgSrc , imgRst , cv::Size(0,0) , ratio , ratio );
	cv::imshow( windowName , imgRst );

//	cvNamedWindow(windowName.c_str());
//	cvShowImage(windowName.c_str(), imgRst);
	if (waitingTime == 0)
		//		cvWaitKey();
		cv::waitKey();
	else
		//cvWaitKey(waitingTime);
		cv::waitKey(waitingTime);

//	cvReleaseImage(&imgRst);
}

void showImage( const IplImage * imgSrc , const float ratio , const string & windowName , const int waitingTime )
{
	int width = int(imgSrc->width  * ratio);
	int heigh = int(imgSrc->height * ratio);

	IplImage * imgRst = cvCreateImage( cvSize( width , heigh ) , imgSrc->depth , imgSrc->nChannels );
	cvResize( imgSrc , imgRst );

	cvNamedWindow( windowName.c_str() );
	cvShowImage( windowName.c_str() , imgRst );
	if ( waitingTime == 0 )
		cvWaitKey();
	else
		cvWaitKey( waitingTime );

	cvReleaseImage( &imgRst );
}

void relu( float * data ,  const int N )
{
	for ( int i = 0 ; i < N ; ++i )
		if ( data[i] < 0 )
			data[i] = 0.;
}

void pooling( float *before , float *after , const int channel , const int width , const int height )
{
	int ah = height/2;
	int aw = width/2;

	for ( int ic = channel ; ic-- ; before += width*height , after += ah *aw )
	for ( int ih = 0 ; ih < ah ; ++ ih )
	for ( int iw = 0 ; iw < aw ; ++ iw )
	{
		after[ ih * aw + iw ] = before[ 2 * ih * width + 2 * iw ];
		
		if ( after[ ih * aw + iw ] < before[ 2 * ih * width + 2 * iw + 1 ] )
			after[ ih * aw + iw ] = before[ 2 * ih * width + 2 * iw + 1 ];

		if ( after[ ih * aw + iw ] < before[ (2 * ih + 1) * width + 2 * iw ] )
			after[ ih * aw + iw ] = before[ (2 * ih + 1) * width + 2 * iw ];

		if ( after[ ih * aw + iw ] < before[ (2 * ih + 1) * width + 2 * iw + 1] )
			after[ ih * aw + iw ] = before[ (2 * ih + 1) * width + 2 * iw + 1];
	}
}

template <>
void wrapper_cblas_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M, const int N, const float alpha,
							const float* A, const float* x, const float beta, float* y)
{
	cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1); 
}

template <>
void wrapper_cblas_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,const int N, const double alpha, 
							const double* A, const double* x, const double beta, double* y)
{
	cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1); 
}

template<>
void wrapper_cblas_gemm<float>( const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
								const int M, const int N, const int K, const float alpha, 
								const float* A, const float* B, const float beta, float* C)
{
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;

	cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template<>
void wrapper_cblas_gemm<double>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
								const int M, const int N, const int K, const double alpha, 
								const double* A, const double* B, const double beta, double* C)
{
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;
	cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

inline bool is_a_ge_zero_and_a_lt_b(int a, int b)
{
	return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void im2col( const Dtype* data_im, const int channels, const int height, 
                const int width, const int kernel_h, const int kernel_w, Dtype* data_col )
{
	const int output_h = height - kernel_h + 1;
	const int output_w = width  - kernel_w + 1;
	const int channel_size = height * width;

	for (int channel = channels; channel--; data_im += channel_size)
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++)
	for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++)
	{
		int input_row = kernel_row;
		for (int output_rows = output_h; output_rows; output_rows--)
		{
			int input_col = kernel_col;
			for (int output_col = output_w; output_col; output_col--)
			{
				*(data_col++) = data_im[input_row * width + input_col];
				input_col ++;
			}
			input_row ++;
		}
	}
}

template void im2col( const float* data_im, const int channels, const int height,
		                const int width, const int kernel_h, const int kernel_w, float* data_col );

bool sort_area( const pair<CvRect , double > & feature1 , const pair<CvRect , double> & feature2 )
{
    return feature1.second > feature2.second;
}

bool sort_rect_area_pair_x( const pair<CvRect , double > & feature1 , const pair<CvRect , double> & feature2 )
{
    return feature1.first.x > feature2.first.x;
}
bool sort_rect_area_pair_y( const pair<CvRect , double > & feature1 , const pair<CvRect , double> & feature2 )
{
    return feature1.first.y > feature2.first.y;
}
void merging_box( CvRect & BBOX , const CvRect & ABOX )
{
	int ori_left = BBOX.x;
	int ori_right = BBOX.x + BBOX.width;
	int ori_up = BBOX.y;
	int ori_down = BBOX.y + BBOX.height;

	int add_left = ABOX.x;
	int add_right = ABOX.x + ABOX.width;
	int add_up = ABOX.y;
	int add_down = ABOX.y + ABOX.height;

	int mer_left = ori_left;
	int mer_right = ori_right;
	int mer_up = ori_up;
	int mer_down = ori_down;
	if ( mer_left > add_left ) mer_left = add_left;
	if ( mer_right < add_right ) mer_right = add_right;
	if ( mer_up > add_up ) mer_up = add_up;
	if ( mer_down < add_down ) mer_down = add_down;

	BBOX.x = mer_left;
	BBOX.y = mer_up;
	BBOX.width = mer_right - mer_left;
	BBOX.height = mer_down - mer_up;
}

void matrix_inversion_2d( MAT2D & a , MAT2D & inverse )
{
	/*
     * 2d matrix inversion has a close form, 
     * but user should be confirmed that the determinant is not zero
     */
    if ( fabs( a.a00 * a.a11 - a.a10 * a.a01 ) < 1E-10 )
    {
//        std::cout << "the determinant of the matrix equals to zero!\n" << std::endl;
        return;
    }
    float inv_det = 1./( a.a00 * a.a11 - a.a10 * a.a01 );

    inverse.a00 =  a.a11 * inv_det;
    inverse.a11 =  a.a00 * inv_det;
    inverse.a10 = -a.a10 * inv_det;
    inverse.a01 = -a.a01 * inv_det;
}

float prior_pro_2d( pair<float , float> & x , pair<float, float> & miu , MAT2D & sigma )
{
    MAT2D sigma_inverse;
    matrix_inversion_2d( sigma , sigma_inverse );

    float det = sigma.a00 * sigma.a11 - sigma.a10 * sigma.a01;

    pair<float , float> tmp;
    // temporary vaiable for calculation (x-miu) * sigma_inverse *(x - miu)

    tmp.first  = ( x.first - miu.first ) * sigma_inverse.a00 + ( x.second - miu.second ) * sigma_inverse.a10 ;
    tmp.second = ( x.first - miu.first ) * sigma_inverse.a01 + ( x.second - miu.second ) * sigma_inverse.a11 ;

    float exp_index = -0.5 * ( tmp.first * ( x.first - miu.first ) + tmp.second * ( x.second - miu.second ) );

    return ( exp( exp_index ) / ( 2. * 3.1415926 * sqrt(det) ) );
}

void feature_exp( vector< pair<float , float> > & features , pair<float , float> & miu )
{
	miu.first = 0.;
	miu.second= 0.;

	for ( int iDATA = 0 ; iDATA < features.size() ; ++ iDATA )
    {
        miu.first += features[iDATA].first;
        miu.second+= features[iDATA].second;
    }
	miu.first /= features.size();
	miu.second/= features.size();
}

void feature_cov( vector< pair<float , float> > & features , MAT2D & sigma )
{
	sigma.a00 = 0.;
	sigma.a01 = 0.;
	sigma.a10 = 0.;
	sigma.a11 = 0.;
	pair<float , float> miu;

	feature_exp( features , miu );

	for ( int iDATA = 0 ; iDATA < features.size() ; ++ iDATA )
    {
        sigma.a00 += ( features[iDATA].first - miu.first ) * ( features[iDATA].first - miu.first );
        sigma.a11 += ( features[iDATA].second- miu.second) * ( features[iDATA].second- miu.second);
        sigma.a01 += ( features[iDATA].first - miu.first ) * ( features[iDATA].second- miu.second);
        sigma.a10 += ( features[iDATA].first - miu.first ) * ( features[iDATA].second- miu.second);
    }
    sigma.a00 /= features.size();
    sigma.a11 /= features.size();
    sigma.a01 /= features.size();
    sigma.a10 /= features.size();
}

template <typename T>
int findMax( vector<T> & score )
{
    T max_ = score[0];
    int index_ = 0;
    for ( int i = 1 ; i < score.size() ; ++ i )
        if ( max_ < score[i] )
        {
            max_ = score[i];
            index_ = i;
        }
    return index_;
}

template int findMax( vector<int> & score );
template int findMax( vector<float> & score );

void compute_score( IplImage * imgSrc , ::caffe::NetParameter & net , vector<float> & score )
{
    score.resize(10);

	float * imgData		= new float [ 28 * 28 ];
	float * k1			= new float [ 20 * 25 ];
	float * b1			= new float [ 20 ];
	float * imgData_col = new float [ 24 * 24 * 25 ];
	float * conv1_col	= new float [ 20 * 24 * 24 ];
	float * pool1_col	= new float [ 20 * 12 * 12 ];

	for ( int irow = 0 ; irow < 28 ; ++ irow )
	for ( int icol = 0 ; icol < 28 ; ++ icol )
	{
		imgData[ irow * 28 + icol ] = cvGetReal2D( imgSrc , irow , icol ) * 0.00390625;
	}

	im2col( imgData , 1 , 28, 28 , 5 , 5 , imgData_col );

	for ( int i = 0 ; i < 20 * 25 ; ++i )
		k1[i] = net.layer(1).blobs(0).data(i);

	for ( int i = 0 ; i < 20 ; ++i )
		b1[i] =  net.layer(1).blobs(1).data(i);

	wrapper_cblas_gemm<float>( CblasNoTrans , CblasNoTrans , 20 ,  24*24 , 25 , 1., k1 ,imgData_col , 0. , conv1_col );

	for ( int i = 0 ; i < 20 * 24 * 24 ; ++ i )
	{
		conv1_col[i] += b1[i/(24*24)];
	}

	pooling( conv1_col , pool1_col , 20 , 24 , 24 );

	float * pool1_2_mat	= new float [ 8 * 8 * 20 * 5 * 5 ];
	float * k2			= new float [ 50 * 20 * 5 * 5 ];
	float * b2			= new float [ 50 ];
	float * conv2_col	= new float [ 50 * 8 * 8 ];
	float * pool2_col	= new float [ 50 * 4 * 4 ];

	im2col( pool1_col , 20 , 12 , 12 , 5 , 5 , pool1_2_mat );

	for ( int i = 0 ; i < 50 * 20 * 5 * 5 ; ++i )
		k2[i] = net.layer(3).blobs(0).data(i);
	for ( int i = 0 ; i < 50 ; i ++ )
		b2[i] = net.layer(3).blobs(1).data(i);

	wrapper_cblas_gemm<float>( CblasNoTrans , CblasNoTrans , 50, 8*8 , 20*5*5, 1., k2 ,pool1_2_mat, 0. , conv2_col );

	for ( int i =0 ; i < 50 * 8 * 8 ; i ++ )
		conv2_col[i] += b2[i/(8*8)];

	pooling( conv2_col , pool2_col , 50 , 8 , 8 );

	float * inner_w1	= new float [ 500 * 800 ];
	float * inner_b1	= new float [ 500 ];
	float * inner_r1	= new float [ 500 ];
	float * inner_w2	= new float [ 10 * 500 ];
	float * inner_b2	= new float [ 10 ];
	float * inner_r2	= new float [ 10 ];

	for ( int i = 0 ; i < 500 * 800 ; i++ )
		inner_w1[i] = net.layer(5).blobs(0).data(i);
	for ( int i = 0 ; i < 500 ; i ++ )
		inner_b1[i] = net.layer(5).blobs(1).data(i);

	wrapper_cblas_gemv<float>( CblasNoTrans , 500 ,800,1.,inner_w1, pool2_col,0,inner_r1);

	for ( int i = 0 ; i < 500 ; i ++ )
		inner_r1[i] += inner_b1[i];

	relu( inner_r1 , 500 );

	for ( int i = 0 ; i < 10 * 500 ; i ++ )
		inner_w2[i] = net.layer(7).blobs(0).data(i);
	for ( int i = 0 ; i < 10 ; i ++ )
		inner_b2[i] = net.layer(7).blobs(1).data(i);
	wrapper_cblas_gemv<float>( CblasNoTrans ,10 ,500,1.,inner_w2, inner_r1,0,inner_r2 );
	for ( int i = 0 ; i < 10 ; i ++ )
		inner_r2[i] += inner_b2[i];

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
	delete [] inner_r2;

}

