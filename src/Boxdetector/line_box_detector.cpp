#include "line_box_detector.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "../util.hpp"

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
	vector<int> ModuleP( WIDTH );
	vector<int> ModuleV( HEIGH );

	for ( int i = 0 ; i < WIDTH ; ++i )
		ModuleP[i] = scoreP[i].first + abs( scoreP[i].second );

	for ( int i = 0 ; i < HEIGH ; ++i )
		ModuleV[i] = scoreV[i].second + abs( scoreV[i].first );

	int maxIndexP = findMax( ModuleP );
	int maxIndexV = findMax( ModuleV );
	// find the first line in x and y direction, respectively.

	for ( int i = 0 ; i < WIDTH ; ++i )
		if ( i > maxIndexP - spaceGap_ && i < maxIndexP + spaceGap_ )
			ModuleP[i] = 0;

	for ( int i = 0 ; i < HEIGH ; ++i )
		if ( i > maxIndexV - spaceGap_ && i < maxIndexV + spaceGap_ )
			ModuleV[i] = 0;

	int maxIndexP2 = findMax( ModuleP );
	int maxIndexV2 = findMax( ModuleV );
	// find another line in x and y direction, respectively.

	int leftLine = maxIndexP >maxIndexP2 ? maxIndexP2:maxIndexP;
	int rightLine = maxIndexP >maxIndexP2 ? maxIndexP:maxIndexP2;
	int upLine = maxIndexV > maxIndexV2 ?maxIndexV2:maxIndexV;
	int downLine = maxIndexV > maxIndexV2 ?maxIndexV:maxIndexV2;

	boxRegion_ = cvRect( leftLine , upLine , rightLine-leftLine , downLine-upLine );

	cvReleaseImage( &imgGray );
	cvReleaseMat( &SobelX );
	cvReleaseMat( &SobelY );
}

void LineBoxDetector::showOnImage() const
{
	IplImage * imgShow = cvCloneImage( imgSrc_ );

	cvRectangle( imgShow , cvPoint( boxRegion_.x , boxRegion_.y) , 
			cvPoint( boxRegion_.x + boxRegion_.width , boxRegion_.y + boxRegion_.height ) , cvScalar(255) );

	cvNamedWindow("show" , CV_WINDOW_NORMAL );
	cvShowImage("show" , imgShow);
	cvWaitKey();

	cvReleaseImage( &imgShow );
}
