#include "tools_classifier.hpp"
#include <utility>
#include <cmath>
#include "util.hpp"
#include "windows.h"

namespace jh
{
	void train_classifier( const vector<IplImage *> imgs , Binarizator & BINTOR , const float & epsilon 
			, int iteration , mg_classifier & mgc )
	{
		vector< pair<float , float> > features;

		int img_num = imgs.size();
		vector<IplImage *> imgs_thres( img_num );

		for ( int i = 0 ; i < img_num ; i++ )
		{
			imgs_thres[i] = cvCreateImage( cvGetSize( imgs[i] ) , 8 , 1 );
			BINTOR.binarizate( imgs[i] , imgs_thres[i] );

			for ( int irow = 0 ; irow < imgs[i]->height ; ++ irow )
			{
				unsigned char * ptr = (unsigned char *) ( imgs[i]->imageData + irow * imgs[i]->widthStep );
				for ( int icol = 0 ; icol < imgs[i]->width ; ++ icol )
				{
					if ( cvGetReal2D( imgs_thres[i] , irow , icol ) == 255 )
						continue;

					int CBlu = ptr[ 3 * icol + 0 ];
					int CGre = ptr[ 3 * icol + 1 ];
					int CRed = ptr[ 3 * icol + 2 ];

					features.push_back( make_pair( (CRed+epsilon) / (CGre+epsilon), (CRed+epsilon) / (CBlu+epsilon)) );
				}
			}
		}
		// get all the feature pts in imgs

		//pair<float , float> miu;
		//MAT2D cov, cov_inv;
		///*
		// * calculate the original expectation and covariance, then 
		// * filter out all the points beyond 99.7% range 
		// * because when we're calculating LIKELIHOOD function, too small probability value
		// * will cause extremly large value
		// */

		//feature_exp( features , miu );
		//feature_cov( features , cov );
		//matrix_inversion_2d( cov , cov_inv );

		//int SIZE = features.size();

		//for ( int iDATA = 0 ; iDATA < SIZE ; ++ iDATA )
		//{
		//	float A = features[iDATA].first - miu.first;
		//	float B = features[iDATA].second- miu.second;

		//	if ( cov_inv.a00 * A * A + 2* cov_inv.a10 *A *B + cov_inv.a11 * B *B < 9. )
		//		features.push_back( features[iDATA] );
		//	/*
		//	 * if sqrt ( (x - u )T * sigma_inv * (x-u) ) < 3
		//	 * then it's filtered out
		//	 */
		//}
		//features.erase( features.begin() , features.begin() + SIZE );
		//// we get the filtered feature pts in img_

		mgc.initExtractor( features );
		mgc.EMAlgorithm( iteration );
		/*
		 * train a RedPixelsExtractor to classify the data points
		 */
		for ( vector<IplImage *>::iterator iter = imgs_thres.begin() ; iter != imgs_thres.end() ; ++iter )
			cvReleaseImage( &(*iter) );
	}

	bool getRedPixels( IplImage * imgSrc , Binarizator & BINTOR , classifier & clf , float epsilon ,
			float prc , float prc_inside_point,  IplImage * imgRst )
	{
		IplImage * img_ = cvCloneImage( imgSrc );
		IplImage * imgThreshold = cvCreateImage( cvGetSize( imgSrc ) , 8 , 1 );
		BINTOR.binarizate( imgSrc , imgThreshold );

		// set pixels around the edges of img4Contour as 0!
		for ( int irow = 0 ; irow < imgThreshold->height ; ++ irow )
		for ( int icol = 0 ; icol < imgThreshold->width ; ++ icol )
		{
			if ( irow <5 || icol < 5 || irow > imgThreshold->height -5 || icol > imgThreshold->width -5 )
				cvSetReal2D( imgThreshold , irow , icol , 255 );
		}

	//	showImage( imgThreshold , 1 , "threshold"  );
		for ( int irow = 0 ; irow < imgSrc->height ; ++ irow )
		for ( int icol = 0 ; icol < imgSrc->width  ; ++ icol )
		{
			if ( cvGetReal2D( imgThreshold , irow , icol ) == 255 )
				cvSet2D( img_ , irow , icol , cvScalar(0,0,0) );
		}
	//	showImage( img_ , 1 , "img"  );

		IplImage * imgColorPixels = cvCreateImage( cvGetSize( img_ ) , 8 , 1 );
		
		cvSetZero( imgColorPixels );
		// the image stores the red pixels , but in gray scale, 
		// after that , we resize the image and get 28 *28 image
		IplImage * img4Contour = cvCreateImage( cvGetSize( img_ ) , 8 , 1 );
		cvSetZero( img4Contour );
		// prepare for creating contours by using cvFindContours;

		int tpt_count = 0;
		// whole pts number; including color pixels and grey pixels
		int pt_count = 0;
		// count the color pixels number;
		for ( int irow = 0 ; irow < img_->height ; ++ irow )
		{
			unsigned char * ptr = (unsigned char *) ( img_->imageData + irow * img_->widthStep );
			unsigned char * ptrC= (unsigned char *) ( imgColorPixels->imageData + irow * imgColorPixels->widthStep );
			unsigned char * ptr4= (unsigned char *) ( img4Contour->imageData + irow * img4Contour->widthStep );
			for ( int icol = 0 ; icol < img_->width ; ++ icol )
			{
				int CBlu = ptr[ 3 * icol + 0 ];
				int CGre = ptr[ 3 * icol + 1 ];
				int CRed = ptr[ 3 * icol + 2 ];

				if ( 0 == CBlu && 0 == CGre && 0 == CRed )
					continue;

				tpt_count ++;

				float feature1 = ( CRed + epsilon ) / ( CGre + epsilon ) ;
				float feature2 = ( CRed + epsilon ) / ( CBlu + epsilon ) ;

				pair<float , float> tmp = make_pair( feature1 , feature2 );

				if ( !clf.isGray(tmp) )
				{
					pt_count ++;
					//ptrC[icol] = ( CRed * 30 + CGre * 59 + CBlu * 11 + 50 )/100;
					ptrC[icol] = 255;
					ptr4[icol] = 255;
				}
			}
		}

		//showImage( imgColorPixels , 1 , "rst" , 500 );

		if ( float(pt_count) / tpt_count < prc )
			return false;

		float xHeart = 0 , yHeart = 0 , xStd = 0 , yStd = 0;

		for ( int irow = 0 ; irow < imgColorPixels->height ; ++ irow )
		{
			unsigned char * ptr = (unsigned char *) ( imgColorPixels->imageData + irow * imgColorPixels->widthStep );
			for ( int icol = 0 ; icol < imgColorPixels->width ; ++ icol )
			{
				if ( ptr[icol] != 0 )
				{
					xHeart += float(icol)/pt_count;
					yHeart += float(irow)/pt_count;
				}
			}
		}

		for ( int irow = 0 ; irow < imgColorPixels->height ; ++ irow )
		{
			unsigned char * ptr = (unsigned char *) ( imgColorPixels->imageData + irow * imgColorPixels->widthStep );
			for ( int icol = 0 ; icol < imgColorPixels->width ; ++ icol )
			{
				if ( ptr[icol] != 0 )
				{
					xStd += 1./pt_count * (icol - xHeart) * (icol -xHeart);
					yStd += 1./pt_count * (irow - yHeart) * (irow -yHeart);
				}
			}
		}
		xStd = sqrt( xStd );
		yStd = sqrt( yStd );

		/*
		 * compute the heart of the color pixels
		 * used to filter contours,
		 * the contours are filtered out using distance-area
		 * area multiplied a coefficient related with distance to the heart
		 * the further it's stay away from the heart, the area will be 
		 * squeezed more
		 */

		CvMemStorage * storage = cvCreateMemStorage();
		CvSeq * contours;
		cvFindContours( img4Contour , storage , &contours , sizeof(CvContour) , CV_RETR_TREE , CV_CHAIN_APPROX_SIMPLE );
		cvSetZero( img4Contour );
		// the image after finding contour has broken, so here it's been set with zero value;
		CvContour * contourGetter = (CvContour *)contours;

		vector< pair<CvRect , double > > contour_feature;

//		MessageBoxA(NULL,"66","66",MB_OK|MB_SYSTEMMODAL);
		if ( contourGetter->rect.width == img4Contour->width - 2 && 
				contourGetter->rect.height == img4Contour->height - 2 )
			contourGetter = ( CvContour * )contourGetter->v_next;
		/*
		 * if the first contour's size is exactly the image size-2 in two dimenstion
		 * that means we get an outter contour contains the whole image
		 * then we need to delete it first, and see the v_next point
		 */
//		MessageBoxA(NULL,"77","77",MB_OK|MB_SYSTEMMODAL);
		do
		{
			double area = fabs (cvContourArea( contourGetter ) );

			float xCenter = contourGetter->rect.x + contourGetter->rect.width / 2;
			float yCenter = contourGetter->rect.y + contourGetter->rect.height / 2;
			// the coordinates of the center of the contour;

			float xPro = std::exp( - (xCenter-xHeart) *(xCenter-xHeart) / (2. *xStd *xStd) ) / (sqrt(2*3.1415926) * xStd);
			float yPro = std::exp( - (yCenter-yHeart) *(yCenter-yHeart) / (2. *yStd *yStd) ) / (sqrt(2*3.1415926) * yStd);
			// the distance weight in x and y direction;

			pair<CvRect , double> tmp = make_pair( contourGetter->rect , area * xPro * yPro );
			contour_feature.push_back( tmp );

			contourGetter = (CvContour *)contourGetter->h_next;
		}
		while( contourGetter != 0 );

		sort( contour_feature.begin() , contour_feature.end() , sort_area );

		double whole_area = 0.;
		for ( int i = 0 ; i < contour_feature.size() ; ++ i )
			whole_area += contour_feature[i].second;

		int box_up , box_do , box_ri , box_le;
		/*
		 * calculate the bounding box's four edges,
		 * if 95 % area lies in the box, then it's qualified
		 * we search through the max area to the min area one by one
		 * when the adding area reaches 95% , we stop
		 */
		box_up = contour_feature[0].first.y ;
		box_do = contour_feature[0].first.y + contour_feature[0].first.height ;
		box_le = contour_feature[0].first.x ;
		box_ri = contour_feature[0].first.x + contour_feature[0].first.width ;

		double adding_area = contour_feature[0].second;
		int index = 1;

		while( adding_area < prc_inside_point * whole_area )
		{
			adding_area += contour_feature[index].second;

			if ( box_up > contour_feature[index].first.y )
				box_up = contour_feature[index].first.y;

			if ( box_do < contour_feature[index].first.y + contour_feature[index].first.height )
				box_do = contour_feature[index].first.y + contour_feature[index].first.height ;

			if ( box_le > contour_feature[index].first.x )
				box_le = contour_feature[index].first.x;

			if ( box_ri < contour_feature[index].first.x + contour_feature[index].first.width )
				box_ri = contour_feature[index].first.x + contour_feature[index].first.width;

			index ++;
		}

		int box_width = box_ri - box_le;
		int box_heigh = box_do - box_up;

		float scale = 20 / float(box_width>box_heigh?box_width:box_heigh);

		int tmp_width = int(28 / scale);
		int tmp_heigh = int(28 / scale);
		IplImage * imgTMP = cvCreateImage( cvSize( tmp_width , tmp_heigh ) , 8 , 1 );
		cvSetZero( imgTMP );
		// this is a temporary image contain the image cut from imgColorPixels
		// which contain at least 95 % area..

		CvRect roi;

		roi.x = box_le;
		roi.y = box_up;
		roi.width = box_width;
		roi.height= box_heigh;

		cvSetImageROI( imgColorPixels , roi );
		int xGap = ( tmp_width - box_width ) / 2;
		int yGap = ( tmp_heigh - box_heigh ) / 2;

		for ( int irow = 0 ; irow < box_heigh ; ++ irow )
			for ( int icol = 0 ; icol < box_width ; ++ icol )
				cvSetReal2D( imgTMP , irow + yGap , icol + xGap , cvGetReal2D( imgColorPixels , irow , icol ) );

		IplImage * imgTMP2 = cvCreateImage( cvSize( 28 , 28 ) , 8 , 1 );
		cvSetZero( imgTMP2 );
		// the image used for containing the resized image from imgTMP;

		cvResize( imgTMP , imgTMP2 , CV_INTER_AREA );

		int xHeart_resize = 0;
		int yHeart_resize = 0;
		int count_resize = 0;

		for ( int irow = 0 ; irow < 28 ; ++ irow )
			for ( int icol = 0 ; icol < 28 ; ++ icol )
			{
				if ( cvGetReal2D( imgTMP2 , irow , icol ) != 0 )
				{
					xHeart_resize += icol;
					yHeart_resize += irow;
					count_resize ++;
				}
			}                                                                                                       
		xHeart_resize /= count_resize;
		yHeart_resize /= count_resize;

		/*
		 * recompute the heart of the resized image
		 * to move the center of the mass to the center
		 * of the image of 28 * 28 size
		 */
		cvSetZero( imgRst );

		for ( int irow = 0 ; irow < 28 ; ++ irow )
			for ( int icol = 0 ; icol < 28 ; ++ icol )
			{
				int xcor = icol - 14 + xHeart_resize;
				int ycor = irow - 14 + yHeart_resize;
				if ( xcor >= 0 && xcor < 28 && ycor >= 0 && ycor < 28 )
					cvSetReal2D( imgRst , irow , icol , cvGetReal2D( imgTMP2 , ycor , xcor ) );
			}

		// test start
		
		for ( int irow = 0 ; irow < 28 ; ++ irow )
			for ( int icol = 0 ; icol < 28 ; ++ icol )
				if ( cvGetReal2D( imgRst , irow , icol ) != 0 )
					cvSetReal2D( imgRst , irow , icol , 255);
		
		// test end
		/**
		 * seems the problem of number "9" can't be
		 * identified has been fixed by setting 
		 * all non-zero value to 255...
		 * i suppose that's not a good way to solve this
		 * but right now we gonna live with it
		 */

//		showImage( imgRst , 10 , "rst" );

		cvReleaseImage( &img_ );
		cvReleaseImage( &imgThreshold );
		cvReleaseImage( &imgColorPixels );
		cvReleaseImage( &img4Contour );
		cvReleaseMemStorage( &storage );
		cvReleaseImage( &imgTMP );
		cvReleaseImage( &imgTMP2);

		return true;
	}

}// namespace jh
