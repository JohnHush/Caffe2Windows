#ifndef __JohnHush_UTIL_H
#define __JohnHush_UTIL_H

#include "caffe/proto/caffe.pb.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cblas.h>
#include <string>
#include "caffe/caffe.hpp"

using std::string;
using std::vector;
using std::pair;

void showImage( const IplImage * imgSrc , const float ratio , const string & windowName , const int waitingTime = 0 );

typedef struct MAT2D
{
	float a00;
	float a01;
	float a10;
	float a11;

}MAT2D;

void relu( float * data , const int N );
void pooling( float *before , float *after , const int channel , const int width , const int height );

template <typename Dtype>
void wrapper_cblas_gemv(const CBLAS_TRANSPOSE TransA, const int M,const int N, const Dtype alpha,
                            const Dtype* A, const Dtype* x, const Dtype beta, Dtype* y);
template <typename Dtype>
void wrapper_cblas_gemm( const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, 
				const int K, const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta, Dtype* C);

template <typename Dtype>
void im2col(const Dtype* data_im, const int channels, const int height, 
				const int width, const int kernel_h, const int kernel_w, Dtype* data_col);

void compute_score( IplImage * imgSrc , ::caffe::NetParameter & net , vector<float> & score );

template <typename T>
int findMax( vector<T> & score );

void matrix_inversion_2d( MAT2D & a , MAT2D & inverse );
float prior_pro_2d( pair<float , float> & x , pair<float, float> & miu , MAT2D & sigma );
void feature_exp( vector< pair<float , float> > & features , pair<float , float> & miu );
void feature_cov( vector< pair<float , float> > & features , MAT2D & sigma );

bool sort_area( const pair<CvRect , double > & feature1 , const pair<CvRect , double> & feature2 );
bool sort_rect_area_pair_x( const pair<CvRect , double > & feature1 , const pair<CvRect , double> & feature2 );
bool sort_rect_area_pair_y( const pair<CvRect , double > & feature1 , const pair<CvRect , double> & feature2 );
void merging_box( CvRect & BBOX , const CvRect & ABOX );
// merging the rectangular box from BBOX, adding ABOX

vector<float> compute_score_by_caffe( const IplImage * imgSrc , const  string & deploy_model , const  string & caffe_model );
// compute the score using caffe library, use its forward() function in Net

void finetune_by_caffe( const string & pretrained_model , const string & train_net_arch_prototxt , const IplImage *     imgSrc , const int label );
// finetune a pre-trained model using caffe lib
//
void getback_to_ORIGINAL_MODEL( const string & pretrained_model , const string & ori_model );

//in case the trained model is unable to use because of bad data.
//we could use this function to get back to original model pretrained by JohnHush

#endif
