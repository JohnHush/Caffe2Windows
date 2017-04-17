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

using std::string;
using std::vector;
using std::pair;

OCRAPI vector<float> compute_score_by_caffe( const IplImage * imgSrc , const  string & deploy_model , const  string & caffe_model );
// compute the score using caffe library, use its forward() function in Net

OCRAPI void finetune_by_caffe( const string & pretrained_model , const string & train_net_arch_prototxt , const IplImage *     imgSrc , const int label );
// finetune a pre-trained model using caffe lib

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
		void addSomeData( vector<IplImage *> & imgAdds , vector<int> & labels );
		void resetDB();
		void showLastData();
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
		void addData( IplImage * imgAdd , int label , int key_index );
		static int BASE_NUMBER_;
};

#endif
