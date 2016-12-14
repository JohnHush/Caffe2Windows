#ifndef UTIL_H
#define UTIL_H

#include "RedPixelsExtractor.hpp"
#include "Blob.hpp"
#include "caffe.pb.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

typedef struct MAT2D
{
	float a00;
	float a01;
	float a10;
	float a11;

}MAT2D;

void conv1_gemm( IplImage * imgSrc , WeightBlob & weight , BiasBlob & bias , WeightBlob & conv1_result );
void max_pooling( WeightBlob & bottom , WeightBlob & up );
void conv2_gemm( WeightBlob & bottom , WeightBlob & weight , BiasBlob & bias , WeightBlob & conv2_result );
void transfer2matrix( WeightBlob & bottom , MatrixBlob & up );
void inner_product( MatrixBlob & bottom , MatrixBlob & weight , BiasBlob & bias , MatrixBlob & inner1_result );
void relu( MatrixBlob & bottom );
void compute_score( IplImage * imgSrc , ::caffe::NetParameter & net , vector<float> & score );
int findMax( vector<float> & score );

void matrix_inversion_2d( MAT2D & a , MAT2D & inverse );
float prior_pro_2d( pair<float , float> & x , pair<float, float> & miu , MAT2D & sigma );
void feature_exp( vector< pair<float , float> > & features , pair<float , float> & miu );
void feature_cov( vector< pair<float , float> > & features , MAT2D & sigma );

bool hasRedPixelsAndPickUp( IplImage * imgSrc , IplImage * imgRst , pair<float, float> & MODEL_PRIOR , const \
								float & epsilon = 100. );

bool sort_area( const pair<CvRect , double > & feature1 , const pair<CvRect , double> & feature2 );
/*
 * the function detect if there are any red pixels in the input image
 * the theory is depicted as follow:
 *
 * first: clustering the points after cvAdaptiveThreshold into 2 clusters
 *        while assuming there are 2 clusters ( red point and black point)
 *        int the image, and then calculate the likelihood function of this
 *        model (the model means: 2 clusters ), and assuming the a prior 
 *        probability of the model is 0.5 because I suppose 1 cluster and 
 *        2 cluster are almost appear in the same probability, basically
 *
 * second: cluster the points after cvAdaptiveThreshold into 1 cluster
 *         so this is simply calculate the expectation and covariance
 *         matrix, then also calculate its likelihood function
 *
 * third: compare the two likelihood function, choose the model has bigger
 *        likelihood function.
 *
 * ( BE CAUTION ): in calculating likelihood function, the features are filtered
 * before calculating the probability.
 * because if p = 0 , then logp = -inf, it will cause problem,
 * so we filtered out 0.3% points which beyond, 3 * std range.
 *
 *
 * the input imgSrc suppose to be 3 channels, with background color is white(255,255,255)
 * and black color and maybe with red color
 *
 * if there is 1 class only in feature points, then return imgRst all 0 values;
 * if there is 2 class in feature points, then return the color pixels in gray scale.
 */

#endif
