#include <caffe/caffe.hpp>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <caffe/proto/caffe.pb.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include "util_caffe.hpp"
#include "util.hpp"

using namespace std;
int main( int argc , char ** argv )
{
	string jpg_path("F:/Lenet_training_set/filtered_with_gaussian_kernel");
	string db_path("F:/Lenet_training_set/filtered_with_gaussian_kernel/db");
	string db_train_path("F:/Lenet_training_set/filtered_with_gaussian_kernel/training_set");
	string db_test_path("F:/Lenet_training_set/filtered_with_gaussian_kernel/testing_set");

	vector<string> db_path_string(1);
	db_path_string[0] = db_path;

	read_Windows_Data2_LevelDB(jpg_path, db_path);
	merge_data_and_split(db_path_string , db_train_path , db_test_path );

	finetune_with_Existing_LevelDB("lenet_FINETUNE.caffemodel", "lenet_train_leveldb.prototxt", db_train_path,
		db_test_path, 15000);

	char s;
	cin >> s;
	return 1;
}
