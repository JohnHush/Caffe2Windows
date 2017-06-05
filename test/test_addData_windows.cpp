#define NOMINMAX
#define NO_STRICT
#include "HandWritingDigitsRecognitionSystem.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include "util.hpp"
#include <stdio.h>
#include "util_caffe.hpp"
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <glog/logging.h>
#include <caffe/proto/caffe.pb.h>
#include <sstream>
#include <caffe/caffe.hpp>
#include <unistd.h>

using std::cout;
using std::endl;


int main( int argc , char * argv[] )
{
	char *file_path;
	file_path = (char*)malloc(128);
	getcwd( file_path , 128 );
	caffe::NetParameter netP;
	ReadProtoFromTextFileOrDie( string(file_path) + string("/CIFAR10_.prototxt") , &netP );

	caffe::Net<float> net(netP);

	return 1;


	initPredictor();
#ifdef _WINDOWS
	IplImage * imgSrc = cvLoadImage( "C:\\handwriting\\20170421\\before\\3__0.34__4564838.jpg" , CV_LOAD_IMAGE_COLOR );
#endif
#ifdef UNIX
	IplImage * imgSrc = cvLoadImage( "/home/pitaloveu/Caffe2Windows/test_data/color_5.jpg" , CV_LOAD_IMAGE_COLOR );
#endif

	//showImage( imgSrc , 1 , "original" , 1000 );
	float confidence;
	IplImage * imgOut = cvCreateImage( cvSize(280 , 280) , 8 , 1 );
	cvSetZero( imgOut );
	
	int score = looksLikeNumber( imgSrc,   imgOut,   confidence , 0.05);

	std::cout << "  score = " << score << std::endl;
	std::cout << "  confidence = " << confidence << std::endl;

	cvReleaseImage( &imgSrc );
//	cvReleaseImage(&imgOut);

	deletePredictor();
#ifdef _WINDOWS
	ldb_handler HANDLER("D:\\MyProjects\\orion-eye\\base_data\\finetune_training_data_leveldb");
	read_Windows_Data2_Existing_LevelDB("C:\\Users\\JohnHush\\Desktop\\finetune_data",  HANDLER );
#endif
#ifdef UNIX
	ldb_handler HANDLER("/home/pitaloveu/Caffe2Windows/finetune_data_withoutBOX/finetune_training_data_leveldb");
	read_Windows_Data2_Existing_LevelDB("/home/pitaloveu/Desktop/finetune_data",  HANDLER );
#endif
	HANDLER.showLastData();
#ifdef _WINDOWS
	HANDLER.splitDB(string("D:\\MyProjects\\orion-eye\\base_data\\TRAINING_FINETUNE"),
		string("D:\\MyProjects\\orion-eye\\base_data\\TESTING_FINETUNE"));
#endif
#ifdef UNIX
	string TRAIN = string("/home/pitaloveu/Desktop/TRAINING_FINETUNE");
	string TEST  = string("/home/pitaloveu/Desktop/TESTING_FINETUNE");
	HANDLER.splitDB( TRAIN , TEST );
#endif
	HANDLER.resetDB();
	HANDLER.closeDB();

	getback_to_ORIGINAL_MODEL("lenet_FINETUNE.caffemodel" , "lenet_ORIGINAL.caffemodel" );
	finetune_with_Existing_LevelDB("lenet_FINETUNE.caffemodel", "lenet_FINETUNE.prototxt");

	char s;

	std::cin >> s;

	return 1;
}
