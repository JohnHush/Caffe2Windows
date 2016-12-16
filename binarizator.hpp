#ifndef BINARIZATOR_H_
#define BINARIZATOR_H_

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

class Binarizator
{
	public:
		Binarizator(){}
		virtual ~Binarizator(){}
		virtual void binarizate( IplImage *imgSrc , IplImage *imgRst ) = 0;
};

#endif
