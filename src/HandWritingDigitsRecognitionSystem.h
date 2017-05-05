#ifndef __JohnHush_HANDWRITINGDIGITSRECOGNITIONSYSTEM_H
#define __JohnHush_HANDWRITINGDIGITSRECOGNITIONSYSTEM_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "config.hpp"

using std::vector;

OCRAPI void initPredictor( int BLOCK_SIZE=201 , double OFFSET=20 );
OCRAPI void deletePredictor( );
OCRAPI int looksLikeNumber( IplImage * imgSrc, IplImage* imgOut , float & confidence , float red_pts_prec = 0.1 );

#endif
