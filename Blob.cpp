
#include "Blob.hpp"

BiasBlob::BiasBlob( const int length )
{
    length_ = length;
    data_   = new float [length_];
}

BiasBlob::BiasBlob( const ::caffe::BlobProto & blob )
{
    if ( blob.shape().dim_size() != 1 )
    {
        ::std::cout << "WRONG DIMENSION SIZE IN FUNCTION BiasBlob( caffe::Blo...)!\n "<< ::std::endl;
        return;
    }
    length_ = blob.shape().dim(0);

    data_ = new float [length_];

    for( int i = 0 ; i < length_ ; ++ i )
        data_[i] = blob.data(i);
}
WeightBlob::WeightBlob( const float * data_in , const int N , const int C , const int H , const int W )
{
    shape_.clear();
    shape_.resize(4);

    shape_[0] = N;
    shape_[1] = C;
    shape_[2] = H;
    shape_[3] = W;

    count_ = shape_[0] * shape_[1] * shape_[2] * shape_[3];

	data_ = new float *** [ shape_[0] ];

    for( int iNUM = 0 ; iNUM < shape_[0] ; ++ iNUM )
    {
        data_[iNUM] = new float ** [ shape_[1] ];
        for( int iCHA = 0 ; iCHA < shape_[1] ; ++ iCHA )
        {
            data_[iNUM][iCHA] = new float * [ shape_[2] ];
            for( int iHEI = 0 ; iHEI < shape_[2] ; ++ iHEI )
            {
                data_[iNUM][iCHA][iHEI] = new float [ shape_[3] ];
            }
        }
    }

    for( int iNUM = 0 ; iNUM < shape_[0] ; ++ iNUM )
    for( int iCHA = 0 ; iCHA < shape_[1] ; ++ iCHA )
    for( int iHEI = 0 ; iHEI < shape_[2] ; ++ iHEI )
    for( int iWID = 0 ; iWID < shape_[3] ; ++ iWID )
        data_[iNUM][iCHA][iHEI][iWID] = data_in[ iNUM * shape_[1] *shape_[2] * shape_[3] + \
                                iCHA * shape_[2] * shape_[3] + iHEI * shape_[3] + iWID ];

}
WeightBlob::WeightBlob( const int N , const int C , const int H , const int W )
{
    shape_.clear();
    shape_.resize(4);

    shape_[0] = N;
    shape_[1] = C;
    shape_[2] = H;
    shape_[3] = W;

    count_ = shape_[0] * shape_[1] * shape_[2] * shape_[3];

	data_ = new float *** [ shape_[0] ];

    for( int iNUM = 0 ; iNUM < shape_[0] ; ++ iNUM )
    {
        data_[iNUM] = new float ** [ shape_[1] ];
        for( int iCHA = 0 ; iCHA < shape_[1] ; ++ iCHA )
        {
            data_[iNUM][iCHA] = new float * [ shape_[2] ];
            for( int iHEI = 0 ; iHEI < shape_[2] ; ++ iHEI )
            {
                data_[iNUM][iCHA][iHEI] = new float [ shape_[3] ];
            }
        }
    }

}
WeightBlob::WeightBlob( const ::caffe::BlobProto & blob )
{
    if ( blob.shape().dim_size() != 4 )
    {
        ::std::cout << "WRONG DIMENSION SIZE IN FUNCTION WeightBlob(caffe::BlobProto) !\n" << ::std::endl;
		return;
    }
    shape_.clear();
    shape_.resize(4);

    shape_[0] = blob.shape().dim(0);
    shape_[1] = blob.shape().dim(1);
    shape_[2] = blob.shape().dim(2);
    shape_[3] = blob.shape().dim(3);

    count_ = shape_[0] * shape_[1] * shape_[2] * shape_[3];

    data_ = new float *** [ shape_[0] ];

    for( int iNUM = 0 ; iNUM < shape_[0] ; ++ iNUM )
    {
        data_[iNUM] = new float ** [ shape_[1] ];
        for( int iCHA = 0 ; iCHA < shape_[1] ; ++ iCHA )
        {
            data_[iNUM][iCHA] = new float * [ shape_[2] ];
            for( int iHEI = 0 ; iHEI < shape_[2] ; ++ iHEI )
            {
                data_[iNUM][iCHA][iHEI] = new float [ shape_[3] ];
            }
        }
    }

    for( int iNUM = 0 ; iNUM < shape_[0] ; ++ iNUM )
    for( int iCHA = 0 ; iCHA < shape_[1] ; ++ iCHA )
    for( int iHEI = 0 ; iHEI < shape_[2] ; ++ iHEI )
    for( int iWID = 0 ; iWID < shape_[3] ; ++ iWID )
        data_[iNUM][iCHA][iHEI][iWID] = blob.data( iNUM * shape_[1] *shape_[2] * shape_[3] + \
                                iCHA * shape_[2] * shape_[3] + iHEI * shape_[3] + iWID );
}

WeightBlob::~WeightBlob()
{
    for( int iNUM = 0 ; iNUM < shape_[0] ; ++ iNUM )
    {
        for( int iCHA = 0 ; iCHA < shape_[1] ; ++ iCHA )
        {
            for ( int iHEI = 0 ; iHEI < shape_[2] ; ++ iHEI )
                delete [] data_[iNUM][iCHA][iHEI];

            delete [] data_[iNUM][iCHA];
        }
        delete [] data_[iNUM];
    }
    delete [] data_;
}

MatrixBlob::MatrixBlob( const int N , const int C )
{
	shape_.clear();
	shape_.resize(2);

	shape_[0] = N;
	shape_[1] = C;

	count_ = N * C;

	data_ = new float *[N];

	for( int i = 0 ; i < N ; ++ i )
		data_[i] = new float [C];
}

MatrixBlob::~MatrixBlob()
{
	for ( int i = 0 ; i < shape_[0] ; ++ i )
		delete [] data_[i];

	delete [] data_;
}

MatrixBlob::MatrixBlob( const ::caffe::BlobProto & blob )
{
    if ( blob.shape().dim_size() != 2 )
    {
        ::std::cout << "WRONG DIMENSION SIZE IN FUNCTION MatrixBlob(caffe::BlobProto) !\n" << ::std::endl;
		return;
    }

    shape_.clear();
    shape_.resize(2);

    shape_[0] = blob.shape().dim(0);
    shape_[1] = blob.shape().dim(1);
	
	count_ = shape_[0] * shape_[1];

	data_ = new float * [ shape_[0] ];

	for ( int i = 0 ; i < shape_[0] ; ++ i )
		data_[i] = new float [ shape_[1] ];

	for ( int iNUM = 0 ; iNUM < shape_[0] ; ++ iNUM )
	for ( int iCHA = 0 ; iCHA < shape_[1] ; ++ iCHA )
		data_[iNUM][iCHA] = blob.data( iNUM * shape_[1] + iCHA );
}
