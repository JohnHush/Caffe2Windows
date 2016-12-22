#include "line_box_detector.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using std::vector;
using std::pair;
using std::make_pair;

void LineBoxDetector::detectBox ()
{
	using std::endl;
	using std::cout;

	int WIDTH = imgSrc_->width;
	int HEIGH = imgSrc_->height;

	IplImage * imgGray = cvCreateImage( cvGetSize( imgSrc_ ) , 8 , 1 );
	if ( imgSrc_->nChannels == 3 )
		cvCvtColor( imgSrc_ , imgGray , CV_BGR2GRAY );
	else
		cvCopy( imgSrc_ , imgGray );

	CvMat * SobelX = cvCreateMat( HEIGH , WIDTH , CV_16SC1 );
	CvMat * SobelY = cvCreateMat( HEIGH , WIDTH , CV_16SC1 );

	cvSobel( imgGray , SobelX , 1 , 0 );
	cvSobel( imgGray , SobelY , 0 , 1 );
	/*
	 * compute the first derivative of the whole image
	 * then we add in one line through x or y direction.
	 * be cautious that we actually transfer every vector to position-x or
	 * positive-y direction to avoid there are polar-opposite problem
	 * that could happen in uneven-lightning situation
	 */
	
	vector< pair<int , int> > scoreP( WIDTH , make_pair(0,0) );
	vector< pair<int , int> > scoreV( HEIGH , make_pair(0,0) );
	// add up Parallelly or Vertically

	for ( int irow = 0 ; irow < HEIGH ; ++ irow )
	{
		short * ptrX = (short *)( SobelX->data.ptr + irow * SobelX->step );
		short * ptrY = (short *)( SobelY->data.ptr + irow * SobelY->step );

		for( int icol = 0 ; icol < WIDTH ; ++ icol )
		{
			if ( ptrX[icol] > 0 )
			{
				scoreP[icol].first  += ptrX[icol];
				scoreP[icol].second += ptrY[icol];
			}
			else
			{
				scoreP[icol].first  += -ptrX[icol];
				scoreP[icol].second += -ptrY[icol];
			}
			if ( ptrY[icol] > 0 )
			{
				scoreV[irow].second += ptrY[icol];
				scoreV[irow].first  += ptrX[icol];
			}
			else
			{
				scoreV[irow].second += -ptrY[icol];
				scoreV[irow].first  += -ptrX[icol];
			}
		}
	}
	sort( scoreP.begin() , scoreP.end() , sortT );
	sort( scoreV.begin() , scoreV.end() , sortT );

}
