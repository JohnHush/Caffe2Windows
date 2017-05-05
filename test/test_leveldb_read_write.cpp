#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include "caffe/proto/caffe.pb.h"
#include <glog/logging.h>
#include "util.hpp"
#include "tools_classifier.hpp"
#include "Binarizator/adaptive_threshold.hpp"
#include "util_caffe.hpp"
#include <unistd.h>

using namespace std;
using namespace caffe;

int main( void )
{
//	leveldb::DB * db;
//	leveldb::Options options;
//	leveldb::Status status = leveldb::DB::Open( options , 
//			"/home/pitaloveu/orion-eye/finetune_data_withoutBOX/finetune_training_data_leveldb" , &db );
//	if( !status.ok() )
//		LOG(FATAL) << "something wroing in openning db" << endl;

//	leveldb::Iterator * it = db->NewIterator( leveldb::ReadOptions() );
	int count = 0;
	IplImage * img = cvCreateImage( cvSize(28,28) , 8 , 1 );


	IplImage * imgIn = cvLoadImage( "/home/pitaloveu/orion-eye/test_data/color_2.jpg" , CV_LOAD_IMAGE_COLOR );
	AdaThre adapt_thresholder( 201 , 20 );
	IplImage * imgred = cvCreateImage( cvSize(28,28) , 8 , 1 );
	cvSetZero( imgred );
	bool hasma = jh::getRedPixelsInHSVRange( imgIn , adapt_thresholder , 0.05 , imgred );


	ldb_handler ldbHandler( "/home/pitaloveu/orion-eye/finetune_data_withoutBOX/finetune_training_data_leveldb" );


	vector<IplImage *> IMGS(1);
	vector<int> LABELS(1);

	IMGS[0] = imgred;
	LABELS[0] = 2;

	ldbHandler.addSomeData( IMGS , LABELS );

	ldbHandler.showLastData();
	ldbHandler.closeDB();
	ldbHandler.resetDB();
	ldbHandler.showLastData();
//	return 1;
//	showImage( imgred , 10 , "red" , 30 );


	leveldb::DB * db;
	leveldb::Options options;
	leveldb::Status status = leveldb::DB::Open( options , 
			"/home/pitaloveu/orion-eye/finetune_data_withoutBOX/finetune_training_data_leveldb" , &db );

	// write in one element;

	caffe::Datum datum;
    char * pixels = new char [ 28 * 28 ];

    for ( int irow = 0 ; irow < 28 ; irow ++ )
    for ( int icol = 0 ; icol < 28 ; icol ++ )
        pixels[ irow * 28 + icol ] = cvGetReal2D( imgred , irow , icol );

    datum.set_data( pixels , 28 *28 );
    datum.set_label( 5 );

    char key_cstr[10];
    string value;
    snprintf( key_cstr , 10 , "%08d" , 2001 );
    datum.SerializeToString( &value );
    string keystr( key_cstr );

    leveldb::WriteOptions write_options;
    write_options.sync = true;
    db->Put( write_options , keystr , value );

	delete db;


	leveldb::DB * db2;
	leveldb::Options options2;
	leveldb::Status status2 = leveldb::DB::Open( options2 , 
			"/home/pitaloveu/orion-eye/finetune_data_withoutBOX/finetune_training_data_leveldb" , &db2 );

	leveldb::Iterator* it = db2->NewIterator( leveldb::ReadOptions() );
	it->SeekToLast();
	string keyS = it->key().ToString();
	std::cout << keyS << std::endl;

	delete db2;
	return 1;

//	leveldb::Iterator * it = db->NewIterator( leveldb::ReadOptions() );

	std::cout << it->key().ToString()<< std::endl;
	for( it->SeekToLast() ; it->Valid() ; it->Prev() , count++ )
	{
		Datum datum;
		datum.ParseFromString( it->value().ToString() );

		if( count == 0 )
		{
			for ( int irow = 0 ; irow < 28 ; irow ++ )
			for ( int icol = 0 ; icol < 28 ; icol ++ )
			{
				int   data_index = irow*28 + icol;
				float data_value = static_cast<float>(static_cast<uint8_t>( (datum.data()) [data_index]));
				cvSetReal2D( img , irow , icol , data_value );
			}
			showImage( img , 10 , "test" , 0 );
			printf( "count = %s\n" , it->key().ToString().c_str() );
		}
	}
	cvReleaseImage( &img );

	return 1;
}
