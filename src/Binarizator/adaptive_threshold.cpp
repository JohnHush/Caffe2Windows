#include "adaptive_threshold.hpp"
#include <iostream>

void AdaThre::binarizate( IplImage *imgSrc , IplImage *imgRst )
{
	// leave some space for GLOG to assert the dimension of imgSrc and imgRst
	
	if ( imgSrc->nChannels != 3 && imgSrc->nChannels != 1 )
	{
		std::cout << "channel number in imgSrc wrong size\n" << std::endl;
		return;
	}

	IplImage * imgGray = cvCreateImage( cvGetSize( imgSrc ) , 8 , 1 );
	if ( imgSrc->nChannels == 3 )
		cvCvtColor( imgSrc , imgGray , CV_BGR2GRAY );
	else
		cvCopy( imgSrc , imgGray );

	cvAdaptiveThreshold( imgGray , imgRst , max_value_ , window_type_, threshold_type_ , block_size_ , param1_ );

	cvReleaseImage( &imgGray );
}
