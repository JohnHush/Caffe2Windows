#include "caffe.pb.h"
#include <opencv2/opencv.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

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


void conv1_gemm( IplImage * imgSrc , WeightBlob & weight , BiasBlob & bias , WeightBlob & conv1_result );
void max_pooling( WeightBlob & bottom , WeightBlob & up );
void conv2_gemm( WeightBlob & bottom , WeightBlob & weight , BiasBlob & bias , WeightBlob & conv2_result );
void transfer2matrix( WeightBlob & bottom , MatrixBlob & up );
void inner_product( MatrixBlob & bottom , MatrixBlob & weight , BiasBlob & bias , MatrixBlob & inner1_result );
void relu( MatrixBlob & bottom );
void compute_score( IplImage * imgSrc , ::caffe::NetParameter & net , vector<float> & score );
int findMax( vector<float> & score );

//uint32_t swap_endian( uint32_t val );

int findMax( vector<float> & score )
{
	float max_ = score[0];
	int index_ = 0;
	for ( int i = 1 ; i < score.size() ; ++ i )
		if ( max_ < score[i] )
		{
			max_ = score[i];
			index_ = i;
		}
	return index_;
}

void relu( MatrixBlob & bottom )
{
	for ( int iNUM = 0 ; iNUM < bottom.getShape()[0] ; ++ iNUM )
	for ( int iCHA = 0 ; iCHA < bottom.getShape()[1] ; ++ iCHA )
		if ( bottom.getValue( iNUM , iCHA ) < 0. )
			bottom.setValue( iNUM , iCHA , 0. );
}

void inner_product( MatrixBlob & bottom , MatrixBlob & weight , BiasBlob & bias , MatrixBlob & inner1_result )
{
	for ( int iNUM = 0 ; iNUM < inner1_result.getShape()[0] ; ++ iNUM )
	for ( int iCHA = 0 ; iCHA < inner1_result.getShape()[1] ; ++ iCHA )
	{
		float inner_sum = 0.;
		for ( int iSUBCHA = 0 ; iSUBCHA < weight.getShape()[1] ; ++ iSUBCHA )
			inner_sum += weight.getValue( iCHA , iSUBCHA ) * bottom.getValue( iNUM , iSUBCHA );

		inner_sum += bias.getValue( iCHA );
		inner1_result.setValue( iNUM , iCHA , inner_sum );
	}
}

void transfer2matrix( WeightBlob & bottom , MatrixBlob & up )
{
	for ( int iNUM = 0 ; iNUM < bottom.getShape()[0] ; ++ iNUM )
	for ( int iCHA = 0 ; iCHA < bottom.getShape()[1] ; ++ iCHA )
	for ( int iHEI = 0 ; iHEI < bottom.getShape()[2] ; ++ iHEI )
	for ( int iWID = 0 ; iWID < bottom.getShape()[3] ; ++ iWID )
	{
		up.setValue( iNUM , iCHA * bottom.getShape()[2] * bottom.getShape()[3] + \
						iHEI * bottom.getShape()[3] + iWID , bottom.getValue( iNUM , iCHA , iHEI , iWID ) );
	}
}

void conv2_gemm( WeightBlob & bottom , WeightBlob & weight , BiasBlob & bias , WeightBlob & conv2_result )
{
	int BOT_WIDTH = bottom.getShape()[3];
    int BOT_HEIGH = bottom.getShape()[2];
	int BOT_CHANE = bottom.getShape()[1];
	int BOT_NUMBE = bottom.getShape()[0];

    int WEI_WIDTH = weight.getShape()[3];
    int WEI_HEIGH = weight.getShape()[2];
    int WEI_CHANE = weight.getShape()[1];
    int WEI_NUMBE = weight.getShape()[0];

    int RES_WIDTH = conv2_result.getShape()[3];
    int RES_HEIGH = conv2_result.getShape()[2];
    int RES_CHANE = conv2_result.getShape()[1];
    int RES_NUMBE = conv2_result.getShape()[0];

	for ( int iNUM = 0 ; iNUM < RES_NUMBE ; ++ iNUM )
	for ( int iCHA = 0 ; iCHA < RES_CHANE ; ++ iCHA )
	for ( int iHEI = 0 ; iHEI < RES_HEIGH ; ++ iHEI )
	for ( int iWID = 0 ; iWID < RES_WIDTH ; ++ iWID )
	{
		float conv_sum = 0.;

		for ( int iSUBCHA = 0 ; iSUBCHA < WEI_CHANE ; ++ iSUBCHA )
		for ( int iKerH   = 0 ; iKerH   < WEI_HEIGH ; ++ iKerH )
		for ( int iKerW   = 0 ; iKerW   < WEI_WIDTH ; ++ iKerW )
			conv_sum += weight.getValue( iCHA , iSUBCHA , iKerH , iKerW ) * \
							bottom.getValue( iNUM , iSUBCHA , iHEI + iKerH , iWID + iKerW );

		conv_sum += bias.getValue( iCHA );
		conv2_result.setValue( iNUM , iCHA , iHEI , iWID , conv_sum );
	}
}

void max_pooling( WeightBlob & bottom , WeightBlob & up )
{
	for ( int iNUM = 0 ; iNUM < up.getShape()[0] ; ++ iNUM )
	for ( int iCHA = 0 ; iCHA < up.getShape()[1] ; ++ iCHA )
	for ( int iHEI = 0 ; iHEI < up.getShape()[2] ; ++ iHEI )
	for ( int iWID = 0 ; iWID < up.getShape()[3] ; ++ iWID )
	{
		float max_value = bottom.getValue( iNUM , iCHA , 2 * iHEI , 2 * iWID );

		for ( int iSUBHEI = 2 * iHEI ; iSUBHEI < 2 * iHEI + 2 ; ++ iSUBHEI )
		for ( int iSUBWID = 2 * iWID ; iSUBWID < 2 * iWID + 2 ; ++ iSUBWID )
			if ( max_value < bottom.getValue( iNUM , iCHA , iSUBHEI , iSUBWID ) )
				max_value = bottom.getValue( iNUM , iCHA , iSUBHEI , iSUBWID );

		up.setValue( iNUM , iCHA , iHEI , iWID , max_value );
	}
}

void conv1_gemm( IplImage * imgSrc , WeightBlob & weight , BiasBlob & bias , WeightBlob & conv1_result )
{
	int IMG_WIDTH = imgSrc->width;
	int IMG_HEIGH = imgSrc->height;
	int WEI_WIDTH = weight.getShape()[3];
	int WEI_HEIGH = weight.getShape()[2];
	int RES_WIDTH = conv1_result.getShape()[3];
	int RES_HEIGH = conv1_result.getShape()[2];

	int Kernel_NUM = bias.getLength();

	for ( int iKernel = 0 ; iKernel < Kernel_NUM ; ++ iKernel )
	{
		for ( int iResH = 0 ; iResH < RES_HEIGH ; ++ iResH )
		for ( int iResW = 0 ; iResW < RES_WIDTH ; ++ iResW )
		{
			float conv_sum = 0.;	// tmporary variable for convolution operation
			for ( int iKerH = 0 ; iKerH < WEI_HEIGH ; ++ iKerH )
			for ( int iKerW = 0 ; iKerW < WEI_WIDTH ; ++ iKerW )
			{
				conv_sum += weight.getValue( iKernel , 0 , iKerH , iKerW ) * \
					cvGetReal2D( imgSrc , iResH + iKerH , iResW + iKerW ) * 0.00390625;
			}
			conv_sum += bias.getValue( iKernel );
			conv1_result.setValue( 0 , iKernel , iResH , iResW , conv_sum );
		}
	}
}

void compute_score( IplImage * imgSrc , ::caffe::NetParameter & net , vector<float> & score )
{
	score.resize(10);
	// conve1
	WeightBlob kernel1( net.layer(1).blobs(0) );
	BiasBlob bias1( net.layer(1).blobs(1));
	WeightBlob conv_result( 1 , net.layer(1).blobs(0).shape().dim(0) , 24 , 24 );
	conv1_gemm( imgSrc , kernel1 , bias1 , conv_result );

	//pooling layer 1
	WeightBlob pooling1( 1 , net.layer(1).blobs(0).shape().dim(0) , 12 , 12 );
	max_pooling( conv_result , pooling1 );

	// conv2	
	WeightBlob kernel2( net.layer(3).blobs(0) );
	BiasBlob bias2( net.layer(3).blobs(1) );
	WeightBlob conv2_result( 1 , 50 , 8 , 8 );
	conv2_gemm( pooling1 , kernel2 , bias2 , conv2_result );

	// pooling 2
	WeightBlob pooling2( 1 , 50 , 4 , 4 );
	max_pooling( conv2_result , pooling2 );

	// transfer the shape
	MatrixBlob from_pooling2( 1 , 800 );
	transfer2matrix( pooling2 , from_pooling2 );

	// inner product layer 1
	MatrixBlob inner1_result( 1 , 500 );
	MatrixBlob inner_weight1( net.layer(5).blobs(0) );
	BiasBlob bias3( net.layer(5).blobs(1) );
	inner_product( from_pooling2 , inner_weight1 , bias3 , inner1_result );

	//relu layer
	relu( inner1_result );

	// innner product layer2
	MatrixBlob inner2_result( 1 , 10 );
	MatrixBlob inner_weight2( net.layer(7).blobs(0) );
	BiasBlob bias4( net.layer(7).blobs(1) );
	inner_product( inner1_result , inner_weight2 , bias4 , inner2_result );

	for ( int i = 0 ; i < 10 ; i++ )
		score[i] = inner2_result.getValue( 0 , i );
}

//uint32_t swap_endian( uint32_t val )
//{
//	val = (( val << 8) & 0xFF00FF00 ) | (( val >>8 ) & 0xFF00FF);
//	return ( val << 16 ) | ( val >> 16 );
//}

int main( void )
{
	const char * filename = "lenet_iter_10000.caffemodel";

	caffe::NetParameter net;

	int fd = open( filename , O_RDONLY );

	if ( fd == -1 )
	{
		cout << "File not found :" << filename << endl;
	}
	FileInputStream * input = new FileInputStream( fd );
	
	ZeroCopyInputStream* raw_input = new FileInputStream(fd);  
	CodedInputStream* coded_input = new CodedInputStream(raw_input);  
	coded_input->SetTotalBytesLimit(536870912, 268435456);  
 
	net.ParseFromCodedStream( coded_input ); 

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

	for ( int i = 0 ; i < 10000 ; i ++ )
	{
		cout << "i = " << i << "  ";

    	image_file.read( pixels , 28* 28);
    	label_file.read( &label , 1 );

    	for ( int irow = 0 ; irow < 28 ; irow ++  )
	    {
			unsigned char * ptr = (unsigned char *)( imgSrc->imageData + irow * imgSrc->widthStep );
        	for ( int icol = 0 ; icol < 28 ; icol ++ )
    	        ptr[icol] = pixels[ irow * 28 + icol ];
    	}

		compute_score( imgSrc , net , score );
		if ( findMax( score ) == (int)( label + 0) )
			count ++;

		if (findMax( score) == int( label + 0 ))
		{
			cout << "accuracy = " << 1.*count/(i+1) << endl;;
		}
		cout << " test of output = " << int( label ) << endl;
	}

	delete coded_input;  
	delete raw_input;  
	close(fd);

	return 0;
}
