#ifndef BLOB_H
#define BLOB_H

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


class BiasBlob
{
    private:
        float * data_;
        int length_;

        BiasBlob( const BiasBlob & );
        BiasBlob & operator = ( const BiasBlob & );

    public:
        explicit BiasBlob( const ::caffe::BlobProto & blob );
        explicit BiasBlob( const int length );
        BiasBlob() : data_(NULL) , length_(0){};
        ~BiasBlob(){ delete data_; };

        int getLength(){ return length_; };

        inline float getValue( const int N )
        {
            return data_[N];
        }
        inline void setValue( const int N , float value )
        {
            data_[N] = value;
        }
};

class WeightBlob
{
    private:
        float **** data_;
        vector<int> shape_;
        int count_;

        WeightBlob( const WeightBlob & );
        WeightBlob & operator = ( const WeightBlob & );
    public:
        explicit WeightBlob( const ::caffe::BlobProto & blob );
        explicit WeightBlob( const int N , const int C , const int H , const int W );
        explicit WeightBlob( const float * , const int N , const int C , const int H , const int W );
        WeightBlob():data_(NULL) , count_(0){};
        ~WeightBlob();

        vector<int> getShape(){ return shape_; }

        inline float getValue( const int N , const int C , const int H , const int W )
        {
            return data_[N][C][H][W];
        }
        inline void setValue( const int N , const int C , const int H , const int W , float value )
        {
            data_[N][C][H][W] = value ;
        }
};

class MatrixBlob
{
	private:
		float ** data_;
		vector<int> shape_;
		int count_;

		MatrixBlob( const MatrixBlob & );
		MatrixBlob & operator = ( const MatrixBlob & );
	public:
		explicit MatrixBlob( const int N , const int C );
		explicit MatrixBlob( const float *, const int N , const int C );
		explicit MatrixBlob( const ::caffe::BlobProto & blob );
		MatrixBlob():data_(NULL) , count_(0){};
		~MatrixBlob();

		vector<int> getShape(){ return shape_; }

		inline float getValue( const int N , const int C)
		{
			return data_[N][C];
		}
		inline void setValue ( const int N , const int C , float value )
		{
			data_[N][C] = value;
		}
};

#endif
