#ifndef __JohnHush_TOOLS_CLASSIFIER_H_
#define __JohnHush_TOOLS_CLASSIFIER_H_

#include "Classifier/Mixed_Gaussian_Classifier.hpp"
#include "util.hpp"
#include "binarizator.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "config.hpp"

using std::vector;

namespace jh
{

bool OCRAPI getRedPixelsInHSVRange( IplImage * imgSrc , Binarizator & BINTOR , float red_pts_prec , IplImage * imgRst );
bool OCRAPI hasPixelsInBox( IplImage * imgSrc , Binarizator & BINTOR , int range , float perc );
void OCRAPI train_classifier( const vector<IplImage *> imgs , Binarizator & BINTOR , const float & epsilon 
			, int iteration , mg_classifier & mgc );

bool OCRAPI getRedPixels( IplImage * imgSrc , Binarizator & BINTOR , classifier & clf , float epsilon , 
		            float prc , float prc_inside_point , IplImage * imgRst );

}// namespace jh

#endif
