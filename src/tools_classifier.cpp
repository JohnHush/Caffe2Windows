#include "tools_classifier.hpp"
#include "Boxdetector/line_box_detector.hpp"
#include <utility>
#include <cmath>
#include "util.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>

using std::make_pair;

namespace jh
{

bool getRedPixelsInHSVRange2(IplImage * imgSrc, Binarizator & BINTOR, float red_pts_prec, IplImage * imgRst)
{
		/*
		 * Using some kind of Binarizator to binarize the image
		 * usually choose adaptive thresholding defined in OpenCV
		*/
		IplImage * imgBin = cvCreateImage(cvGetSize(imgSrc), 8, 1);
		IplImage * imgRed = cvCreateImage(cvGetSize(imgSrc), 8, 1);
		IplImage * imgBla = cvCreateImage(cvGetSize(imgSrc), 8, 1);
		IplImage * imgHSV = cvCreateImage(cvGetSize(imgSrc), 8, 3);
		IplImage * imgGra = cvCreateImage(cvGetSize(imgSrc), 8, 1);
		IplImage * imgBla4Dilate = cvCreateImage(cvGetSize(imgSrc), 8, 1);
		BINTOR.binarizate(imgSrc, imgBin);

//		showImage(imgBin, 1, "imgbin", 1000);

		cvCvtColor(imgSrc, imgGra, CV_BGR2GRAY);

		cvSetZero(imgRed);
		cvSetZero(imgBla);
		cvSetZero(imgBla4Dilate);

		/*
		* seperate red points and black pts based on HSV value
		* differences, the H value of red is nearly 0 and
		* meanwhile the S value shouldn't be too low
		*/
		cvCvtColor(imgSrc, imgHSV, CV_BGR2HSV);

		int pts_count = 0;
		for (int irow = 0; irow < imgBin->height; ++irow)
			for (int icol = 0; icol < imgBin->width; ++icol)
			{
				int HSV_H = cvGet2D(imgHSV, irow, icol).val[0];

				if (cvGetReal2D(imgBin, irow, icol) != 255)
				{
					pts_count++;
					if (cvGet2D(imgHSV, irow, icol).val[1] > 50 && (HSV_H < 20 || HSV_H > 160))
						cvSetReal2D(imgRed, irow, icol, 255);
					else
					{
						cvSetReal2D(imgBla, irow, icol, 255);
						cvSetReal2D(imgBla4Dilate, irow, icol, 255);
					}
				}
			}
		IplConvKernel * dilate_kernel1 = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_RECT);
		IplConvKernel * dilate_kernel2 = cvCreateStructuringElementEx(5, 5, 2, 2, CV_SHAPE_RECT);

#ifdef DEBUG
		showImage(imgRed, 1., "red pixels in the image ");
		showImage(imgBla, 1., "black pixels in the image ");
#endif
		/*
		* finding all the lines satisfy the conditons given below
		* using cvHoughLine2,
		* the line length should be larger than 2/3 of min(height ,width) of the image
		* and max_gap = 5;
		*/

		int THRESHOLD_LINE = 2 * std::min(imgBla->width, imgBla->height) / 3;
		int MIN_LINE = 1;
		int MAX_GAP = 5;
		CvMemStorage *storage = cvCreateMemStorage();
		CvSeq *lines = 0;
		lines = cvHoughLines2(imgBla, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI / 180
			, THRESHOLD_LINE, MIN_LINE, MAX_GAP);

		cvSetZero(imgBla);
		
		int LEFT_MOST = imgBla->width - 1;
		int RIGH_MOST = 0;
		int UP_MOST = imgBla->height - 1;
		int DO_MOST = 0;

		for (int i = 0; i<lines->total; i++)
		{
			CvPoint *line = (CvPoint *)cvGetSeqElem(lines, i);

			if (abs(line[0].x - line[1].x) > abs(line[0].y - line[1].y))
			{
				if ((line[0].y + line[1].y) / 2 < UP_MOST)
					UP_MOST = (line[0].y + line[1].y) / 2;
				if ((line[0].y + line[1].y) / 2 > DO_MOST)
					DO_MOST = (line[0].y + line[1].y) / 2;
			}
			else
			{
				if ((line[0].x + line[1].x) / 2 < LEFT_MOST)
					LEFT_MOST = (line[0].x + line[1].x) / 2;
				if ((line[0].x + line[1].x) / 2 > RIGH_MOST)
					RIGH_MOST = (line[0].x + line[1].x) / 2;
			}

			cvLine(imgBla, line[0], line[1], cvScalar(255), 1, CV_AA);
		}

		if (LEFT_MOST > imgBla->width / 3)
			LEFT_MOST = 0;
		if (RIGH_MOST < (imgBla->width * 2) / 3)
			RIGH_MOST = imgBla->width - 1;
		if (UP_MOST > imgBla->height / 3)
			UP_MOST = 0;
		if (DO_MOST < (imgBla->height * 2) / 3)
			DO_MOST = imgBla->height - 1;
		
		cvDilate(imgBla, imgBla, dilate_kernel2);

		for (int ir = 0; ir < imgBla->height; ir++)
			for (int ic = 0; ic < imgBla->width; ic++)
				if (ir < UP_MOST || ir > DO_MOST || ic < LEFT_MOST || ic > RIGH_MOST)
					cvSetReal2D( imgBla , ir , ic , 255 );
//		showImage(imgBla, 1, "black",100);
//		cvDilate(imgBla4Dilate, imgBla4Dilate, dilate_kernel);

		//for (int irow = 0; irow < imgBla->height; ++irow)
		//	for (int icol = 0; icol < imgBla->width; ++icol)
		//		if (cvGetReal2D(imgBla, irow, icol) != 0 )
		//			cvSetReal2D(imgRed, irow, icol, 0);

#ifdef DEBUG
		showImage(imgBla, 1, "the dilated boundary lines");
		showImage(imgRed, 1, "the residual red points after filtered out the dilated boundary lines");
#endif
		
		int red_pts_count = 0;
		int bla_pts_count = 0;
		for (int irow = 0; irow < imgRed->height; ++irow)
			for (int icol = 0; icol < imgRed->width; ++icol)
			{
				if (cvGetReal2D(imgRed, irow, icol) != 0)
					red_pts_count++;
				if (cvGetReal2D(imgBla4Dilate, irow, icol) != 0 && cvGetReal2D(imgBla, irow, icol) ==0)
					bla_pts_count++;
			}
		
		/*
		to conquer the problem of red pixels leaking out of
		black pixels, we erode the red pixels first, then
		do dilate, then thin red pixels will be eleminated.
		*/
		if (1. * bla_pts_count / pts_count > red_pts_prec)
		{
			cvErode(imgRed, imgRed, dilate_kernel1);
			cvDilate(imgRed, imgRed, dilate_kernel1);
		}

		cvReleaseImage(&imgBin);
//		cvReleaseImage(&imgBla);
		cvReleaseImage(&imgBla4Dilate);
		cvReleaseImage(&imgHSV);
		cvReleaseStructuringElement(&dilate_kernel1);
		cvReleaseStructuringElement(&dilate_kernel2);
		
		if (1. * red_pts_count / pts_count < red_pts_prec)
		{
			cvReleaseImage(&imgGra);
			cvReleaseImage(&imgRed);
			cvReleaseImage(&imgBla);
			return false;
		}
		// filter imgGra using imgRed mask
		for (int irow = 0; irow < imgRed->height; ++irow)
			for (int icol = 0; icol < imgRed->width; ++icol)
				if (cvGetReal2D(imgRed, irow, icol) == 0)
					cvSetReal2D(imgGra, irow, icol, 0);

		/*
		* eleminate all the small noises based on the contour area
		* first we extract all the contours in the imgRed
		* then we sort the area from high to low
		* then we add then one by one until we get at least 95% area of the whole area
		* that means we get rid of small ones;
		*/
		CvSeq * contours;
		
		IplImage * imgRedClone = cvCloneImage(imgRed);
		
		cvFindContours(imgRed, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		CvContour * contourGetter = (CvContour *)contours;
		vector< pair<CvRect, double> > contour_rect_area_pair;
		
		do
		{
			CvRect & rect = contourGetter->rect;
			CvPoint pt1 = cvPoint( rect.x , rect.y );
			CvPoint pt2 = cvPoint( rect.x + rect.width , rect.y + rect.height );

			if (cvGetReal2D(imgBla, pt1.y, pt1.x) != 0 && cvGetReal2D(imgBla, pt2.y, pt2.x) != 0)
			{
				contourGetter = (CvContour *)contourGetter->h_next;
				continue;
			}

			pair<CvRect, double> tmp = make_pair(contourGetter->rect, fabs(cvContourArea(contourGetter)));
			contour_rect_area_pair.push_back(tmp);

			contourGetter = (CvContour *)contourGetter->h_next;
		} while (contourGetter != 0);
		
		cvReleaseMemStorage(&storage);
		
		// do the box merging and adding area
		sort(contour_rect_area_pair.begin(), contour_rect_area_pair.end(), sort_area);
		double whole_area = 0.;
		for (int i = 0; i < contour_rect_area_pair.size(); ++i)
			whole_area += contour_rect_area_pair[i].second;

		CvRect BBOX = contour_rect_area_pair[0].first;
		double adding_area = contour_rect_area_pair[0].second;
		int index = 1;

		while (adding_area < 0.92 * whole_area)
		{
			merging_box(BBOX, contour_rect_area_pair[index].first);
			adding_area += contour_rect_area_pair[index].second;
			index++;
		}

		// ready to rsize the gray image in BBOX range
		cvSetImageROI(imgRedClone, BBOX);

		int box_width = BBOX.width;
		int box_heigh = BBOX.height;
		
		// MNIST data is 20 * 20 size but in a box of 28 *28 
		// in this function we convert the destinated pic into 280 * 280 pixels
		float scale = 200 / float(box_width>box_heigh ? box_width : box_heigh);

		int tmp_width = int(280 / scale);
		int tmp_heigh = int(280 / scale);

		IplImage * imgTMP = cvCreateImage(cvSize(tmp_width, tmp_heigh), 8, 1);
		cvSetZero(imgTMP);

		int xGap = (tmp_width - box_width) / 2;
		int yGap = (tmp_heigh - box_heigh) / 2;

		for (int ir = 0; ir < box_heigh; ++ir)
			for (int ic = 0; ic < box_width; ++ic)
				cvSetReal2D(imgTMP, ir + yGap, ic + xGap, cvGetReal2D(imgRedClone, ir, ic));

		IplImage * imgTMP2 = cvCreateImage(cvSize(280, 280), 8, 1);
		cvSetZero(imgTMP2);

		cvResize(imgTMP, imgTMP2, CV_INTER_AREA);

		// MNIST data 's heart in the heart of the image
		int xHeart_resize = 0;
		int yHeart_resize = 0;
		int count_resize = 0;

		for (int ir = 0; ir < 280; ++ir)
			for (int ic = 0; ic < 280; ++ic)
			{
				if (cvGetReal2D(imgTMP2, ir, ic) != 0)
				{
					xHeart_resize += ic;
					yHeart_resize += ir;
					count_resize++;
				}
			}
		xHeart_resize /= count_resize;
		yHeart_resize /= count_resize;

		cvSetZero(imgRst);

		for (int ir = 0; ir < 280; ++ir)
			for (int ic = 0; ic < 280; ++ic)
			{
				int xcor = ic - 140 + xHeart_resize;
				int ycor = ir - 140 + yHeart_resize;
				if (xcor >= 0 && xcor < 280 && ycor >= 0 && ycor < 280)
					cvSetReal2D(imgRst, ir, ic, cvGetReal2D(imgTMP2, ycor, xcor));
			}

		// ususally after resizing ,the gray scale is usually low , so we rescale it to 0-255
		int ave_gray = 0;
		int gray_count = 0;
		for (int ir = 0; ir < 280; ++ir)
			for (int ic = 0; ic < 280; ++ic)
			{
				if (cvGetReal2D(imgRst, ir, ic) != 0)
				{
					gray_count++;
					ave_gray += cvGetReal2D(imgRst, ir, ic);
				}
			}
		ave_gray /= gray_count;

		float scale_factor = 255. / ave_gray;

		
		for (int ir = 0; ir < 280; ++ir)
			for (int ic = 0; ic < 280; ++ic)
			{
				if (int(cvGetReal2D(imgRst, ir, ic) *scale_factor * 0.7) > 255)
					cvSetReal2D(imgRst, ir, ic, 255);
				else
					cvSetReal2D(imgRst, ir, ic, int(cvGetReal2D(imgRst, ir, ic) *scale_factor * 0.8));
			}
		
//		showImage( imgRst , 10 , "dd" );
		cvReleaseImage(&imgTMP);
		cvReleaseImage(&imgTMP2);
		cvReleaseImage(&imgGra);
		cvReleaseImage(&imgRed);
		cvReleaseImage(&imgRedClone);
		cvReleaseImage(&imgBla);

		return true;
}


bool hasPixelsInBox( IplImage * imgSrc , Binarizator & BINTOR , int range , float perc )
{
	IplImage * img_thres = cvCreateImage( cvGetSize( imgSrc ) , 8 , 1 );
	BINTOR.binarizate( imgSrc , img_thres );

	//showImage( img_thres , 1 , "dd" , 0 );

	LineBoxDetector lbd( imgSrc , range );
	lbd.detectBox();
	CvRect box = lbd.getBoxRegion();

	int num_white = 0;
	int num_color = 0;

	for ( int irow = 0  ; irow < img_thres->height; ++irow )
		for ( int icol = 0  ; icol < img_thres->width ; ++icol )
		{
			if ( irow < box.y + 10 || irow > box.y + box.height - 10 || 
					icol < box.x + 10 || icol > box.x + box.width - 10 )
				continue;
			if ( cvGetReal2D( img_thres , irow , icol ) == 255 )
				num_white ++;
			else
				num_color ++;
		}
	cvReleaseImage( &img_thres );

	if ( float(num_color) / num_white > perc )
		return true;
	return false;
}

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

	showImage( imgThreshold , 1 , "threshold"  );
	for ( int irow = 0 ; irow < imgSrc->height ; ++ irow )
		for ( int icol = 0 ; icol < imgSrc->width  ; ++ icol )
		{
			if ( cvGetReal2D( imgThreshold , irow , icol ) == 255 )
				cvSet2D( img_ , irow , icol , cvScalar(0,0,0) );
		}
	//showImage( img_ , 1 , "img"  );

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
				ptrC[icol] = ( CRed * 30 + CGre * 59 + CBlu * 11 + 50 )/100;
				//ptrC[icol] = 255;
				ptr4[icol] = 255;
			}
		}
	}

	//showImage( imgColorPixels , 1 , "rst" , 0 );

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
	showImage( imgColorPixels , 1 , "tt" , 1000 );

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
	/*		
			for ( int irow = 0 ; irow < 28 ; ++ irow )
			for ( int icol = 0 ; icol < 28 ; ++ icol )
			if ( cvGetReal2D( imgRst , irow , icol ) != 0 )
			cvSetReal2D( imgRst , irow , icol , 255);
			*/		
	// test end

	int ave_gray = 0;
	int gray_count = 0;
	for ( int ir = 0 ; ir < 28 ; ++ ir )
		for ( int ic = 0 ; ic < 28 ; ++ ic )
		{
			if ( cvGetReal2D( imgRst , ir , ic ) != 0 )
			{
				gray_count ++;
				ave_gray += cvGetReal2D( imgRst , ir , ic );
			}
		}
	ave_gray /= gray_count;

	float scale_factor = 255./ave_gray;

	for ( int ir = 0 ; ir < 28 ; ++ ir )
		for ( int ic = 0 ; ic < 28 ; ++ ic )
		{
			if ( int(cvGetReal2D( imgRst , ir , ic ) *scale_factor * 0.7) > 255 )
				cvSetReal2D( imgRst , ir , ic , 255 );
			else
				cvSetReal2D( imgRst , ir , ic , int(cvGetReal2D( imgRst , ir , ic ) *scale_factor * 0.8) );
		}

	/**
	 * seems the problem of number "9" can't be
	 * identified has been fixed by setting 
	 * all non-zero value to 255...
	 * i suppose that's not a good way to solve this
	 * but right now we gonna live with it
	 */

	//showImage( imgRst , 10 , "rst" );

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
