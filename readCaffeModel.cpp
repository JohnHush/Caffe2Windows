#include "caffe.pb.h"
#include <opencv2/opencv.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <iostream>
#include <string>
#include <vector>

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

void conv1_gemm( IplImage * imgSrc , WeightBlob & weight , BiasBlob & bias , WeightBlob & conv1_result )
{
	int IMG_WIDTH = imgSrc->width;
	int IMG_HEIGH = imgSrc->height;
	int WEI_WIDTH = weight.getShape()[3];
	int WEI_HEIGH = weight.getShape()[2];
	int RES_WIDTH = conv1_result.getShape()[3];
	int RES_HEIGH = conv1_result.getShape()[2];

	if ( IMG_WIDTH - WEI_WIDTH + 1 != RES_WIDTH || IMG_HEIGH - WEI_HEIGH + 1 != RES_HEIGH )
	{
		::std::cout << "WIDTH OR HEIGHT UNMATCHING!\n" << ::std::endl;
		return;
	}
	if ( weight.getShape()[0] != conv1_result.getShape()[0] || weight.getShape()[0] != bias.getLength() )
	{
		::std::cout << "NUM UNMATCHING!\n" << ::std::endl;
		return;
	}
	if ( weight.getShape()[1] != 1 )
	{
		::std::cout << "WEIGHT'S CHANNEL UNMATCHING!\n" << ::std::endl;
		return;
	}
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
					cvGetReal2D( imgSrc , iResH + iKerH , iResW + iKerW );
			}
			conv_sum += bias.getValue( iKernel );
			conv1_result.setValue( iKernel , 0 , iResH , iResW , conv_sum );
		}
	}
}

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

	WeightBlob kernel1( net.layer(1).blobs(0) );
	BiasBlob bias1( net.layer(1).blobs(1));

	WeightBlob conv_result( net.layer(1).blobs(0).shape().dim(0) , 1 , 24 , 24 );

	IplImage * imgSrc = cvCreateImage( cvSize( 28 , 28 ) , 8 , 1 );
	conv1_gemm( imgSrc , kernel1 , bias1 , conv_result );

	for ( int i = 0 ; i < net.layer_size() ; ++ i )
	{
		cout << "layer name = "<< net.layer(i).name() << endl;
		cout << "blob_size = " << net.layer(i).blobs_size() << endl;
		for( int j = 0 ; j < net.layer(i).blobs_size() ; ++ j )
			cout << "blob_dim = " << net.layer(i).blobs(j).shape().dim_size() << endl;
	}

	delete coded_input;  
	delete raw_input;  
	close(fd);

	return 0;
}
