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

	IplImage * imgSrc = cvLoadImage( "C:/Users/JohnHush/Desktop/wrong_data/5__0.83__19777463.jpg" , CV_LOAD_IMAGE_COLOR );
	IplImage * imgSrc2 = cvLoadImage("C:/Users/JohnHush/Desktop/wrong_data/00000020.jpg", CV_LOAD_IMAGE_GRAYSCALE);

//	showImage( imgSrc , 1 , "original" , 1000 );
	float confidence;
	IplImage * imgOut = cvCreateImage( cvSize(280 , 280) , 8 , 1 );
	IplImage * imgOut2 = cvCreateImage(cvSize(280, 280), 8, 1);
	IplImage * imgTST = cvCreateImage(cvSize(28, 28), 8, 1);
	cvSetZero( imgOut );
	
	int index = looksLikeNumber( imgSrc , imgOut , confidence );

	cvSmooth(imgOut, imgOut2 , 2 , 51);
	for (int irow = 0; irow < 280; ++irow)
		for (int icol = 0; icol < 280; icol++)
			if (cvGetReal2D(imgOut2, irow, icol) >= 0)
				cvSetReal2D( imgOut2 , irow , icol ,  uchar(1.8 * cvGetReal2D( imgOut2 , irow , icol ))  );
	cvResize(imgOut2, imgTST );
	cvResize(imgTST , imgOut2);
	cvResize(imgOut2, imgTST);
	cvResize(imgTST, imgOut2);
	cvResize(imgOut2, imgTST);
	cvResize(imgTST, imgOut2);


	std::cout << "index = " << index << std::endl;
	std::cout << "confidence = " << confidence << std::endl;

	showImage(imgOut2, 1, "after processing", 100);

	cvReleaseImage( &imgSrc );
	cvReleaseImage(&imgOut);

	deletePredictor();

	char s;
	std::cin >>s;

	return 1;
}
