#include <caffe/proto/caffe.pb.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <fcntl.h>
#include "util_caffe.hpp"
#include "Binarizator/adaptive_threshold.hpp"
#include "tools_classifier.hpp"

using namespace std;
//using namespace caffe;

int main( int argc , char ** argv )
{
	//string s1("D:\\MyProjects\\orion-eye\\base_data\\TESTING_FINETUNE");
	string s2("D:\\MyProjects\\orion-eye\\base_data\\TRAINING_FINETUNE");
	//string s3("C:\\Users\\JohnHush\\Desktop\\mnist_train_leveldb");
	//string s4("C:\\Users\\JohnHush\\Desktop\\mnist_test_leveldb");

	rotate_leveldb( s2 , string("C:\\Users\\JohnHush\\Desktop\\s2_ROTATE"), 10 );
	//rotate_leveldb(s3, string("C:\\Users\\JohnHush\\Desktop\\s3_ROTATE"), 3);
	//rotate_leveldb(s4, string("C:\\Users\\JohnHush\\Desktop\\s4_ROTATE"), 3);

	//vector<string> db_paths(1);
	//db_paths[0] = string("C:\\Users\\JohnHush\\Desktop\\s2_ROTATE");
	//db_paths[1] = string("C:\\Users\\JohnHush\\Desktop\\s3_ROTATE");
	//db_paths[2] = string("C:\\Users\\JohnHush\\Desktop\\s4_ROTATE");

	//string TRAINING("C:\\Users\\JohnHush\\Desktop\\TRAINING");
	//string TESTING("C:\\Users\\JohnHush\\Desktop\\TESTING");

	//merge_data_and_split( db_paths , TRAINING , TESTING );

#ifdef _WINDOWS
	string deployModel( "D:\\MyProjects\\orion-eye\\deploy_lenet.prototxt" );
	string caffeModel( "D:\\MyProjects\\orion-eye\\lenet_FINETUNE.caffemodel" );
#endif
	getback_to_ORIGINAL_MODEL("lenet_FINETUNE.caffemodel" , "lenet_ORIGINAL.caffemodel" );
	finetune_with_Existing_LevelDB("lenet_FINETUNE.caffemodel", "lenet_train_leveldb.prototxt");

	char sb;
	std::cin >> sb;
	return 0;
}
