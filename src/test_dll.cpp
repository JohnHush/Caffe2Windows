#include "HandWritingDigitsRecognitionSystem.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <windows.h>

int main( int argc , char * argv[] )
{
	initPredictor();

	IplImage * imgSrc = cvLoadImage( "D:\\MyProjects\\orion-eye\\test_data\\color_5.jpg" , CV_LOAD_IMAGE_COLOR );

	float confidence;

	int index = looksLikeNumber( imgSrc , confidence );

	std::cout << "index = " << index << std::endl;

	char s;

	std::cin >>s;

	cvReleaseImage( &imgSrc );

	deletePredictor();

	return 1;
}
