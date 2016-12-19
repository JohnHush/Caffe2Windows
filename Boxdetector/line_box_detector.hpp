#ifndef LINE_BOX_DETECTOR_H_
#define LINE_BOX_DETECTOR_H_

#include "../boxDetector.hpp"
#include <opencv2/opencv.hpp>

class LineBoxDetector : public BoxDetector
{
	protected:
		float spaceGap_;
		/**
		 * filter out the lines within the gap in both direction
		 * which means we only need some lines don't gather together
		 */
		float angleGap_;
		/*
 		 * the angle distortion the program could tolerate, 
 		 * the max difference between two lines is 0+-angleGap_ or 90 += angleGap_
 		 * which represent parallel or vertical lines, respectively.
 		 */
	public:
		LineBoxDetector( float space_gap , float angle_gap ): spaceGap_(space_gap), 
				angleGap_(angle_gap){}
		~LineBoxDetector(){}
		CvRect detect ( const IplImage * imgSrc );
}

#endif
