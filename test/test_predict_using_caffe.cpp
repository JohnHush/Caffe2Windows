#define NOMINMAX
#define NO_STRICT
#include "HandWritingDigitsRecognitionSystem.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include "util.hpp"
#include <stdio.h>
#include "util_caffe.hpp"
//#include "caffe.pb.h"

//int getAllImages(vector< pair<string, int> > & imgName, string path, int LABEL= -1);
int main( int argc , char * argv[] )
{
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

	vector<pair<string, int> > imgName;

//	int count = getAllImages(  imgName, string("C:\\Users\\JohnHush\\Desktop\\finetune_data")  );

//	printf( "number of images = %d\n" , count  );

	char s;

	std::cin >> s;

	return 1;
}

