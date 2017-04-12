#include "HandWritingDigitsRecognitionSystem.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main( int argc , char * argv[] )
{
	initPredictor();
#ifdef _WINDOWS
	IplImage * imgSrc = cvLoadImage( "D:\\MyProjects\\orion-eye\\test_data\\color_5.jpg" , CV_LOAD_IMAGE_COLOR );
#endif
#ifdef UNIX
	IplImage * imgSrc = cvLoadImage( "/home/pitaloveu/orion-eye/test_data/color_5.jpg" , CV_LOAD_IMAGE_COLOR );
#endif

	float confidence;

	int index = looksLikeNumber( imgSrc , confidence );

	std::cout << "index = " << index << std::endl;
	std::cout << "confidence = " << confidence << std::endl;

	char s;

	std::cin >>s;

	cvReleaseImage( &imgSrc );

	deletePredictor();

	return 1;
}
