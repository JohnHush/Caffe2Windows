#ifndef MIXED_GAUSSIAN_RPD_H_
#define MIXED_GAUSSIAN_RPD_H_

#include "../redPixelDetector.hpp"
#include "../RedPixelsExtractor.hpp"
#include "../util.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class MixedGaussianRPD : public RedPixelDetector
{
	public:
		explicit MixedGaussianRPD( const IplImage * imgSrc , float MP1C = 0.6 , 
				float MP2C = 0.4 , float epsilon=100 ) :RedPixelDetector( imgSrc ) ,
				MODEL_PRIOR_1CLASS_(MP1C) , MODEL_PRIOR_2CLASS_(MP2C) , epsilon_(epsilon){}
		virtual void hasRedPixels();
		virtual void getRedPixels( IplImage * imgRed );
		virtual ~MixedGaussianRPD(){}
		bool redOrNot(){ return hasRed_; }

	protected:
		RedPixelsExtractor RedPtEx_;
		float MODEL_PRIOR_1CLASS_;
		float MODEL_PRIOR_2CLASS_;
		float epsilon_;
		void deNoise( IplImage * imgRst );
	private:
		MixedGaussianRPD( const MixedGaussianRPD & );
		MixedGaussianRPD & operator=( const MixedGaussianRPD & );
};

#endif
