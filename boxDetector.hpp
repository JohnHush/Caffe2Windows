#ifndef BOXDETECTOR_H_
#define BOXDETECTOR_H_

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

/*
 * This class aims to search an box in the image, the searched box could be the resulted 
 * shape of warp transformation of a normal rectangular box,
 * and the boxRegion_ should be the region after we transform back the image
 */

class BoxDetector
{
	public:
		explicit BoxDetector( IplImage * imgSrc ): imgSrc_(imgSrc){}
		virtual ~BoxDetector(){}
		virtual void detectBox() = 0;
		inline CvRect getBoxRegion() const { return boxRegion_; }
	protected:
		CvRect boxRegion_;
		IplImage * imgSrc_;
};

#endif
