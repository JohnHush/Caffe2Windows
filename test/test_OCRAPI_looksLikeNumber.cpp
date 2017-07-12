#include "config.hpp"
#include "HandWritingDigitsRecognitionSystem.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include "util.hpp"
#include "caffe.pb.h"

int main( int argc , char * argv[] )
{
	initPredictor();

	IplImage * imgSrc = cvLoadImage( "C:/Users/JohnHush/Desktop/wrong_data/1__0.61__19754609.jpg" , CV_LOAD_IMAGE_COLOR );
	float confidence;
	
	IplImage * imgOut = cvCreateImage( cvSize(280 , 280) , 8 , 1 );
	
	cvSetZero( imgOut );
	
	int index = looksLikeNumber( imgSrc , imgOut , confidence );

	std::cout << "index = " << index << std::endl;
	std::cout << "confidence = " << confidence << std::endl;

	showImage(imgOut, 1, "after processing",10);

	cvReleaseImage( &imgSrc );
	cvReleaseImage(&imgOut);

	deletePredictor();

	char s;
	std::cin >>s;

	return 1;
}
