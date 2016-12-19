#ifndef BOXDETECTOR_H_
#define BOXDETECTOR_H_

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

class BoxDetector
{
	public:
		BoxDetector(){}
		virtual ~BoxDetector(){}
		virtual CvRect detect( const IplImage *imgSrc ) = 0;
};

#endif
