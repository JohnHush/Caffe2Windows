#ifndef __JohnHush_ADAPTIVE_THRESHOLD_
#define __JohnHush_ADAPTIVE_THRESHOLD_

#include "../binarizator.hpp"
#include <opencv2/opencv.hpp>
#include "../config.hpp"

class OCRAPI AdaThre : public Binarizator
{
	private:
		double max_value_;
		/*
		 * max_value and 0 value are set when
		 * condition fixed, individually 
		 * this is relyed on the parameter threshold_type
		 */
		int threshold_type_;
		/*
 		 * cv::CV_THRESH_BINARY or cv::CV_THRESH_BINARY_INV
 		 */
		int window_type_;
		/*
 		 * cv::CV_ADAPTIVE_THRESH_MEAN_C or CV_ADAPTIVE_THRESH_GAUSSIAN_C 
 		 */
		int block_size_;
		// should be odd positive number
		double param1_;
		/**
 		 * this parameter is minused from the calculated MEAN
 		 */
	public:
		AdaThre( int blocksize=55 , double param1=20 )
			:block_size_(blocksize) , param1_(param1)
		{
			threshold_type_ = CV_THRESH_BINARY;
			window_type_    = CV_ADAPTIVE_THRESH_MEAN_C;
			max_value_		= 255;
		}
		inline void setBlocksize( const int & size ){ block_size_ = size; }
		inline void setTtype( const int & Ttype ){ threshold_type_ = Ttype; }
		inline void setWtype( const int & Wtype ){ window_type_ = Wtype; }
		inline void setParam( const double & Param ){ param1_ = Param; }
		inline void setMaxV( const double & MaxV ){ max_value_ = MaxV; }

		virtual ~AdaThre(){}
		virtual void binarizate( IplImage *imgSrc , IplImage *imgRst );
		/*
 		 * this function is used to binarizate the imgSrc image into
 		 * binary type , it calls cvAdaptiveThreshold method in opencv2
 		 * and it doesn't require input image being 8-bit type
 		 * it will transfer it into 8-bit type if it's not, so don't worry
 		 */
};
#endif
