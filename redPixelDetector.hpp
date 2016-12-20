#ifndef REDPIXELDETECTOR_H_
#define REDPIXELDETECTOR_H_

#include <opencv2/opencv.hpp>

class RedPixelDetector
{
	public:
		explicit RedPixelDetector( const IplImage * imgSrc ):
			img_(imgSrc) , hasRed_(false) {}
		/*
		 * the detector only handles IplImage * data structure defined in OpenCV,
		 * it's just the image the user would like to detect, not the classifier
		 */
		virtual ~RedPixelDetector(){}

		virtual void hasRedPixels() = 0;
		/*
		 * decide whether the image has red pixels or not, it's a pure virtual function
		 * should be redesigned in herichical class, 
		 * we're assuming the input imgSrc is a 3 channel color image, and
		 * the background color is black( CvScalar( 0, 0, 0 ) ), this map is filtered from
		 * a Binarizator, while lines are given color, background is given black color
		 * it's converse from the function hasRedPixelsAndPickUp but it's more descriptive.
		 */
		virtual bool isRed( const CvPoint & pt ){
			return false;
		};
		/*
		 * show whether one point is red or not in the image
		 */
		virtual void getRedPixels( IplImage * imgRed ) = 0;
		/*
		 * obtain the whole image filtered by isRed() function
		 * the imgRed should be 8-bit image with single channel( gray scale )
		 * if the point is gray, it's been set 0, else it's been set 255
		 */
	protected:
		const IplImage * img_;
		bool hasRed_;
	private:
		RedPixelDetector( const RedPixelDetector & );
		RedPixelDetector & operator=( const RedPixelDetector &);
		/*
		 * disable the copy constructor and '=' operator
		 * to avoid some problem.
		 */
};

#endif
