#define NOMINMAX
#define NO_STRICT
#include "HandWritingDigitsRecognitionSystem.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include "util.hpp"
#include "caffe.pb.h"

int main( int argc , char * argv[] )
{
	initPredictor();
#ifdef _WINDOWS
	IplImage * imgSrc2 = cvLoadImage( "C:\\handwriting\\20170418\\ShiBieLvGao\\6\\29267347_p.jpg" , CV_LOAD_IMAGE_GRAYSCALE );
#endif
#ifdef UNIX
	IplImage * imgSrc = cvLoadImage( "/home/pitaloveu/orion-eye/test_data/color_5.jpg" , CV_LOAD_IMAGE_COLOR );
#endif

	
	IplImage * imgSrc = cvCreateImage(cvSize(28, 28), 8, 1);
	
	for (int irow = 0; irow < 28; irow++)
		for (int icol = 0; icol < 28; icol++)
			cvSetReal2D( imgSrc , irow , icol , cvGetReal2D( imgSrc2 , irow*10 , icol*10 ) );

	//showImage( imgSrc , 1 , "original" , 1000 );
//	float confidence;
//	IplImage * imgOut = cvCreateImage( cvSize(280 , 280) , 8 , 1 );
//	cvSetZero( imgOut );
	
	std::fstream input( "D:\\MyProjects\\orion-eye\\lenet_FINETUNE.caffemodel" , std::ios::in | std::ios::binary );
//	int index = looksLikeNumber( imgSrc , imgOut , confidence );

	caffe::NetParameter netP;
	
	netP.ParseFromIstream( &input );
	

	vector<float> score(10);

	compute_score( imgSrc , netP ,  score);
	
	for (int i = 0; i < 10; i++)
		std::cout << "i = " << i << "  score = " << score[i] << std::endl;

//	std::cout << "index = " << index << std::endl;
//	std::cout << "confidence = " << confidence << std::endl;

//	showImage(imgOut, 1, "after processing", 0);
	char s;

	std::cin >>s;

	cvReleaseImage( &imgSrc );
//	cvReleaseImage(&imgOut);

	deletePredictor();

	return 1;
}
