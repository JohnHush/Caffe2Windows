#define NOMINMAX
#define NO_STRICT
#include "HandWritingDigitsRecognitionSystem.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include "util.hpp"
#include <io.h>
#include <stdio.h>
#include "util_caffe.hpp"
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <glog/logging.h>
#include <caffe/proto/caffe.pb.h>
#include <sstream>

using std::cout;
using std::endl;


int main( int argc , char * argv[] )
{
	initPredictor();
#ifdef _WINDOWS
	IplImage * imgSrc = cvLoadImage( "C:\\handwriting\\20170421\\before\\3__0.34__4564838.jpg" , CV_LOAD_IMAGE_COLOR );
#endif
#ifdef UNIX
	IplImage * imgSrc = cvLoadImage( "/home/pitaloveu/orion-eye/test_data/color_5.jpg" , CV_LOAD_IMAGE_COLOR );
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

	ldb_handler HANDLER("D:\\MyProjects\\orion-eye\\base_data\\finetune_training_data_leveldb");
	read_Windows_Data2_Existing_LevelDB("C:\\Users\\JohnHush\\Desktop\\finetune_data",  HANDLER );

	HANDLER.showLastData();
	HANDLER.splitDB(string("D:\\MyProjects\\orion-eye\\base_data\\TRAINING_FINETUNE"),
		string("D:\\MyProjects\\orion-eye\\base_data\\TESTING_FINETUNE"));
	HANDLER.resetDB();
	HANDLER.closeDB();


	char s;

	std::cin >> s;

	return 1;
}