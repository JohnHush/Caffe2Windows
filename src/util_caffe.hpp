#ifndef __JohnHush_UTIL_CAFFE_H
#define __JohnHush_UTIL_CAFFE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cblas.h>
#include <string>

#ifdef BUILD_OCR_PREDICT
#include "caffe.pb.h"
#else
#include <caffe/proto/caffe.pb.h>
#endif

#include "util.hpp"
#include "config.hpp"
#include <leveldb/db.h>
class ldb_handler;

using std::string;
using std::vector;
using std::pair;

#ifdef _WINDOWS
OCRAPI void read_Windows_Data2_LevelDB(string data_path, string lmdb_path);
OCRAPI void read_Windows_Data2_Existing_LevelDB(string data_path, ldb_handler & HANDLER);
OCRAPI int getAllImages(vector< pair<string, int> > & imgName, string path, int LABEL = -1);
#endif

OCRAPI vector<float> compute_score_by_caffe( const IplImage * imgSrc , const  string & deploy_model , const  string & caffe_model );
// compute the score using caffe library, use its forward() function in Net

OCRAPI void finetune_by_caffe( const string & pretrained_model , const string & train_net_arch_prototxt , const IplImage * imgSrc , const int label );
// finetune a pre-trained model using caffe lib

OCRAPI void finetune_by_caffe_leveldb(const string & pretrained_model, const string & train_net_arch_prototxt, 
												vector<cv::Mat> & imgs, vector<int> & labels , const string & base_db );
// finetune a pre-trained model using caffe lib based on a base leveldb;

OCRAPI void finetune_with_Existing_LevelDB( const string & pretrained_model, const string & train_net_arch_prototxt );

// we have processed the training db from labeled WINDOWS images, here just finetune.

OCRAPI void getback_to_ORIGINAL_MODEL( const string & pretrained_model , const string & ori_model );
//in case the trained model is unable to use because of bad data.
//we could use this function to get back to original model pretrained by JohnHush

class OCRAPI ldb_handler
{
	private:
		enum DB_OPEN_OR_NOT{ OPEN , CLOSE };
	public:
		explicit ldb_handler( const string & db_name );
		// the level db should exist or it will cause error
		void closeDB();
		void addSomeData( vector<cv::Mat> & imgAdds , vector<int> & labels );
		void resetDB();
		void showLastData();
		void splitDB( string & training_set , string & test_set );
		// split the database into training set and validation set
	private:
		leveldb::DB * db_;
		// the processed database pointer
		leveldb::Options options_;
		leveldb::WriteOptions write_options_;
		leveldb::Status status_;
		leveldb::Iterator * it_;
		string db_name_;
		DB_OPEN_OR_NOT state_;
		// calculate the input elements' number
		void addData(cv::Mat & imgAdd , int label , int key_index );
		static int BASE_NUMBER_;
};

#endif
