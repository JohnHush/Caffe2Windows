#include "util.hpp"
#include <iostream>
#include <vector>

using std::cout;
using std::endl;

bool sort_area( const pair<CvRect , double > & feature1 , const pair<CvRect , double> & feature2 )
{
    return feature1.second > feature2.second;
}

void matrix_inversion_2d( MAT2D & a , MAT2D & inverse )
{
	/*
     * 2d matrix inversion has a close form, 
     * but user should be confirmed that the determinant is not zero
     */
    if ( fabs( a.a00 * a.a11 - a.a10 * a.a01 ) < 1E-10 )
    {
        std::cout << "the determinant of the matrix equals to zero!\n" << std::endl;
        return;
    }
    float inv_det = 1./( a.a00 * a.a11 - a.a10 * a.a01 );

    inverse.a00 =  a.a11 * inv_det;
    inverse.a11 =  a.a00 * inv_det;
    inverse.a10 = -a.a10 * inv_det;
    inverse.a01 = -a.a01 * inv_det;
}

float prior_pro_2d( pair<float , float> & x , pair<float, float> & miu , MAT2D & sigma )
{
    MAT2D sigma_inverse;
    matrix_inversion_2d( sigma , sigma_inverse );

    float det = sigma.a00 * sigma.a11 - sigma.a10 * sigma.a01;

    pair<float , float> tmp;
    // temporary vaiable for calculation (x-miu) * sigma_inverse *(x - miu)

    tmp.first  = ( x.first - miu.first ) * sigma_inverse.a00 + ( x.second - miu.second ) * sigma_inverse.a10 ;
    tmp.second = ( x.first - miu.first ) * sigma_inverse.a01 + ( x.second - miu.second ) * sigma_inverse.a11 ;

    float exp_index = -0.5 * ( tmp.first * ( x.first - miu.first ) + tmp.second * ( x.second - miu.second ) );

    return ( exp( exp_index ) / ( 2. * 3.1415926 * sqrt(det) ) );
}

void feature_exp( vector< pair<float , float> > & features , pair<float , float> & miu )
{
	miu.first = 0.;
	miu.second= 0.;

	for ( int iDATA = 0 ; iDATA < features.size() ; ++ iDATA )
    {
        miu.first += features[iDATA].first;
        miu.second+= features[iDATA].second;
    }
	miu.first /= features.size();
	miu.second/= features.size();
}

void feature_cov( vector< pair<float , float> > & features , MAT2D & sigma )
{
	sigma.a00 = 0.;
	sigma.a01 = 0.;
	sigma.a10 = 0.;
	sigma.a11 = 0.;
	pair<float , float> miu;

	feature_exp( features , miu );

	for ( int iDATA = 0 ; iDATA < features.size() ; ++ iDATA )
    {
        sigma.a00 += ( features[iDATA].first - miu.first ) * ( features[iDATA].first - miu.first );
        sigma.a11 += ( features[iDATA].second- miu.second) * ( features[iDATA].second- miu.second);
        sigma.a01 += ( features[iDATA].first - miu.first ) * ( features[iDATA].second- miu.second);
        sigma.a10 += ( features[iDATA].first - miu.first ) * ( features[iDATA].second- miu.second);
    }
    sigma.a00 /= features.size();
    sigma.a11 /= features.size();
    sigma.a01 /= features.size();
    sigma.a10 /= features.size();
}

bool hasRedPixelsAndPickUp ( IplImage * imgSrc , IplImage * imgRst , pair<float , float> & MODEL_PRIOR , \
								const float & epsilon )
{
	float MODEL_PRIOR_1CLASS = MODEL_PRIOR.first;
	float MODEL_PRIOR_2CLASS = MODEL_PRIOR.second;

	if ( imgSrc->nChannels != 3 )
	{
		cout << " the image channels in function hasRedPixels is not 3!\n" << endl;
		cvSetZero( imgRst );
		return false;
	}
	vector< pair<float , float> > features;

	for ( int irow = 0 ; irow < imgSrc->height ; ++ irow )
	{
		unsigned char * ptr = (unsigned char *) ( imgSrc->imageData + irow * imgSrc->widthStep );
		for ( int icol = 0 ; icol < imgSrc->width ; ++ icol )
		{
			int CBlu = ptr[ 3 * icol + 0 ];
			int CGre = ptr[ 3 * icol + 1 ];
			int CRed = ptr[ 3 * icol + 2 ];

			if ( 255 == CBlu && 255 == CGre && 255 == CRed )
				continue;

			features.push_back( make_pair( (CRed+epsilon) / (CGre+epsilon) , (CRed+epsilon) / (CBlu+epsilon)) );
		}
	}
	// get all the feature points.

	pair<float , float> exp;
    MAT2D cov, cov_inv;
	// calculate the original expectation and covariance, then 
	// filter out all the points beyond 99.7% range 
	
	feature_exp( features , exp );
	feature_cov( features , cov );
	matrix_inversion_2d( cov , cov_inv );

	int SIZE = features.size();

	for ( int iDATA = 0 ; iDATA < SIZE ; ++ iDATA )
	{
		float A = features[iDATA].first - exp.first;
		float B = features[iDATA].second- exp.second;

		if ( cov_inv.a00 * A * A + 2* cov_inv.a10 *A *B + cov_inv.a11 * B *B < 9. )
			features.push_back( features[iDATA] );
		/*
 		 * if sqrt ( (x - u )T * sigma_inv * (x-u) ) < 3
 		 * then it's filtered out
 		 */
	}
	features.erase( features.begin() , features.begin() + SIZE );

	RedPixelsExtractor classifier;
	classifier.initExtractor( features );
	classifier.EMAlgorithm();
	/*
	 * train a redPixelsExtractor to classify the data points
	 */

	float LIKELIHOOD_2CLASS = 0., LIKELIHOOD_1CLASS = 0.;

	LIKELIHOOD_2CLASS = classifier.relativeLikelihood( MODEL_PRIOR_2CLASS );
	
	pair<float , float> exp_1class = make_pair( 0. , 0. );
	MAT2D cov_1class;

	feature_exp( features , exp_1class );
	feature_cov( features , cov_1class );

	for ( int iDATA = 0 ; iDATA < features.size() ; ++ iDATA )
	{
		LIKELIHOOD_1CLASS += log( MODEL_PRIOR_1CLASS * prior_pro_2d(features[iDATA] , exp_1class , cov_1class ) );
	}
	if ( LIKELIHOOD_1CLASS > LIKELIHOOD_2CLASS )
	{
		cvSetZero( imgRst );
		return false;
	}
	// if it's class 1, then it finds no color pixels

	IplImage * imgColorPixels = cvCreateImage( cvGetSize( imgSrc ) , 8 , 1 );
	cvSetZero( imgColorPixels );
	// the image stores the red pixels , but in gray scale, 
	// after that , we resize the image and get 28 *28 image
	IplImage * img4Contour = cvCreateImage( cvGetSize( imgSrc ) , 8 , 1 );
	cvSetZero( img4Contour );
	// prepare for creating contours by using cvFindContours;

	int pt_count = 0;
	// count the color pixels number;
	for ( int irow = 0 ; irow < imgSrc->height ; ++ irow )
    {
        unsigned char * ptr = (unsigned char *) ( imgSrc->imageData + irow * imgSrc->widthStep );
        unsigned char * ptrC= (unsigned char *) ( imgColorPixels->imageData + irow * imgColorPixels->widthStep );
        unsigned char * ptr4= (unsigned char *) ( img4Contour->imageData + irow * img4Contour->widthStep );
        for ( int icol = 0 ; icol < imgSrc->width ; ++ icol )
        {
            int CBlu = ptr[ 3 * icol + 0 ];
            int CGre = ptr[ 3 * icol + 1 ];
            int CRed = ptr[ 3 * icol + 2 ];

            if ( 255 == CBlu && 255 == CGre && 255 == CRed )
                continue;

			float feature1 = ( CRed + epsilon ) / ( CGre + epsilon ) ; 
			float feature2 = ( CRed + epsilon ) / ( CBlu + epsilon ) ; 

			pair<float , float> tmp = make_pair( feature1 , feature2 );

			if ( !classifier.isGray(tmp) )
			{
				pt_count ++;
				//ptrC[icol] = ( CRed * 30 + CGre * 59 + CBlu * 11 + 50 )/100;
				ptrC[icol] = 255;
				ptr4[icol] = 255;
			}
        }
    }

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

	if ( contourGetter->rect.width == img4Contour->width - 2 && 
			contourGetter->rect.height == img4Contour->height - 2 )
		contourGetter = ( CvContour * )contourGetter->v_next;
	/*
 	 * if the first contour's size is exactly the image size-2 in two dimenstion
 	 * that means we get an outter contour contains the whole image
 	 * then we need to delete it first, and see the v_next point
 	 */
	do
	{
		double area = fabs (cvContourArea( contourGetter ) );

		float xCenter = contourGetter->rect.x + contourGetter->rect.width / 2;
		float yCenter = contourGetter->rect.y + contourGetter->rect.height / 2;
		// the coordinates of the center of the contour;

		float xPro = ::exp( - (xCenter-xHeart) *(xCenter-xHeart) / (2. *xStd *xStd) ) / (sqrt(2*3.1415926) * xStd);
		float yPro = ::exp( - (yCenter-yHeart) *(yCenter-yHeart) / (2. *yStd *yStd) ) / (sqrt(2*3.1415926) * yStd);
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
     * if 99.7 % area lies in the box, then it's qualified
     * we search through the max area to the min area one by one
     * when the adding area reaches 99.7% , we stop
     */
	box_up = contour_feature[0].first.y ;
	box_do = contour_feature[0].first.y + contour_feature[0].first.height ;
	box_le = contour_feature[0].first.x ;
	box_ri = contour_feature[0].first.x + contour_feature[0].first.width ;

	double adding_area = contour_feature[0].second;
	int index = 1;

	while( adding_area < 0.997 * whole_area )
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
	// which contain 99.7 % area..
	
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

	cvNamedWindow( "show", CV_WINDOW_NORMAL );
	cvShowImage("show", imgRst);
	cvWaitKey();

	cvReleaseImage( &imgColorPixels );
	cvReleaseImage( &img4Contour );
	cvReleaseMemStorage( &storage );
	cvReleaseImage( &imgTMP );
	cvReleaseImage( &imgTMP2);

	return true;
}

template <typename T>
int findMax( vector<T> & score )
{
    T max_ = score[0];
    int index_ = 0;
    for ( int i = 1 ; i < score.size() ; ++ i )
        if ( max_ < score[i] )
        {
            max_ = score[i];
            index_ = i;
        }
    return index_;
}

template int findMax( vector<int> & score );

void relu( MatrixBlob & bottom )
{
    for ( int iNUM = 0 ; iNUM < bottom.getShape()[0] ; ++ iNUM )
    for ( int iCHA = 0 ; iCHA < bottom.getShape()[1] ; ++ iCHA )
        if ( bottom.getValue( iNUM , iCHA ) < 0. )
            bottom.setValue( iNUM , iCHA , 0. );
}

void inner_product( MatrixBlob & bottom , MatrixBlob & weight , BiasBlob & bias , MatrixBlob & inner1_result )
{
    for ( int iNUM = 0 ; iNUM < inner1_result.getShape()[0] ; ++ iNUM )
    for ( int iCHA = 0 ; iCHA < inner1_result.getShape()[1] ; ++ iCHA )
    {
        float inner_sum = 0.;
        for ( int iSUBCHA = 0 ; iSUBCHA < weight.getShape()[1] ; ++ iSUBCHA )
            inner_sum += weight.getValue( iCHA , iSUBCHA ) * bottom.getValue( iNUM , iSUBCHA );

        inner_sum += bias.getValue( iCHA );
        inner1_result.setValue( iNUM , iCHA , inner_sum );
    }
}

void transfer2matrix( WeightBlob & bottom , MatrixBlob & up )
{
    for ( int iNUM = 0 ; iNUM < bottom.getShape()[0] ; ++ iNUM )
    for ( int iCHA = 0 ; iCHA < bottom.getShape()[1] ; ++ iCHA )
    for ( int iHEI = 0 ; iHEI < bottom.getShape()[2] ; ++ iHEI )
    for ( int iWID = 0 ; iWID < bottom.getShape()[3] ; ++ iWID )
    {
        up.setValue( iNUM , iCHA * bottom.getShape()[2] * bottom.getShape()[3] + \
                        iHEI * bottom.getShape()[3] + iWID , bottom.getValue( iNUM , iCHA , iHEI , iWID ) );
    }
}

void conv2_gemm( WeightBlob & bottom , WeightBlob & weight , BiasBlob & bias , WeightBlob & conv2_result )
{
    int BOT_WIDTH = bottom.getShape()[3];
    int BOT_HEIGH = bottom.getShape()[2];
    int BOT_CHANE = bottom.getShape()[1];
    int BOT_NUMBE = bottom.getShape()[0];

    int WEI_WIDTH = weight.getShape()[3];
    int WEI_HEIGH = weight.getShape()[2];
    int WEI_CHANE = weight.getShape()[1];
    int WEI_NUMBE = weight.getShape()[0];

    int RES_WIDTH = conv2_result.getShape()[3];
    int RES_HEIGH = conv2_result.getShape()[2];
    int RES_CHANE = conv2_result.getShape()[1];
    int RES_NUMBE = conv2_result.getShape()[0];

    for ( int iNUM = 0 ; iNUM < RES_NUMBE ; ++ iNUM )
    for ( int iCHA = 0 ; iCHA < RES_CHANE ; ++ iCHA )
    for ( int iHEI = 0 ; iHEI < RES_HEIGH ; ++ iHEI )
    for ( int iWID = 0 ; iWID < RES_WIDTH ; ++ iWID )
    {
        float conv_sum = 0.;

        for ( int iSUBCHA = 0 ; iSUBCHA < WEI_CHANE ; ++ iSUBCHA )
        for ( int iKerH   = 0 ; iKerH   < WEI_HEIGH ; ++ iKerH )
        for ( int iKerW   = 0 ; iKerW   < WEI_WIDTH ; ++ iKerW )
            conv_sum += weight.getValue( iCHA , iSUBCHA , iKerH , iKerW ) * \
                            bottom.getValue( iNUM , iSUBCHA , iHEI + iKerH , iWID + iKerW );

        conv_sum += bias.getValue( iCHA );
        conv2_result.setValue( iNUM , iCHA , iHEI , iWID , conv_sum );
    }
}

void max_pooling( WeightBlob & bottom , WeightBlob & up )
{
    for ( int iNUM = 0 ; iNUM < up.getShape()[0] ; ++ iNUM )
    for ( int iCHA = 0 ; iCHA < up.getShape()[1] ; ++ iCHA )
    for ( int iHEI = 0 ; iHEI < up.getShape()[2] ; ++ iHEI )
    for ( int iWID = 0 ; iWID < up.getShape()[3] ; ++ iWID )
    {
        float max_value = bottom.getValue( iNUM , iCHA , 2 * iHEI , 2 * iWID );

        for ( int iSUBHEI = 2 * iHEI ; iSUBHEI < 2 * iHEI + 2 ; ++ iSUBHEI )
        for ( int iSUBWID = 2 * iWID ; iSUBWID < 2 * iWID + 2 ; ++ iSUBWID )
            if ( max_value < bottom.getValue( iNUM , iCHA , iSUBHEI , iSUBWID ) )
                max_value = bottom.getValue( iNUM , iCHA , iSUBHEI , iSUBWID );

        up.setValue( iNUM , iCHA , iHEI , iWID , max_value );
    }
}

void conv1_gemm( IplImage * imgSrc , WeightBlob & weight , BiasBlob & bias , WeightBlob & conv1_result )
{
    int IMG_WIDTH = imgSrc->width;
    int IMG_HEIGH = imgSrc->height;
    int WEI_WIDTH = weight.getShape()[3];
    int WEI_HEIGH = weight.getShape()[2];
    int RES_WIDTH = conv1_result.getShape()[3];
    int RES_HEIGH = conv1_result.getShape()[2];

    int Kernel_NUM = bias.getLength();

    for ( int iKernel = 0 ; iKernel < Kernel_NUM ; ++ iKernel )
    {
        for ( int iResH = 0 ; iResH < RES_HEIGH ; ++ iResH )
        for ( int iResW = 0 ; iResW < RES_WIDTH ; ++ iResW )
        {
            float conv_sum = 0.;    // tmporary variable for convolution operation
            for ( int iKerH = 0 ; iKerH < WEI_HEIGH ; ++ iKerH )
            for ( int iKerW = 0 ; iKerW < WEI_WIDTH ; ++ iKerW )
            {
                conv_sum += weight.getValue( iKernel , 0 , iKerH , iKerW ) * \
                    cvGetReal2D( imgSrc , iResH + iKerH , iResW + iKerW ) * 0.00390625;
            }
            conv_sum += bias.getValue( iKernel );
            conv1_result.setValue( 0 , iKernel , iResH , iResW , conv_sum );
        }
    }
}

void compute_score( IplImage * imgSrc , ::caffe::NetParameter & net , vector<float> & score )
{
    score.resize(10);
    // conve1
    WeightBlob kernel1( net.layer(1).blobs(0) );
    BiasBlob bias1( net.layer(1).blobs(1));
    WeightBlob conv_result( 1 , net.layer(1).blobs(0).shape().dim(0) , 24 , 24 );
    conv1_gemm( imgSrc , kernel1 , bias1 , conv_result );

    //pooling layer 1
    WeightBlob pooling1( 1 , net.layer(1).blobs(0).shape().dim(0) , 12 , 12 );
    max_pooling( conv_result , pooling1 );

    // conv2    
    WeightBlob kernel2( net.layer(3).blobs(0) );
    BiasBlob bias2( net.layer(3).blobs(1) );
    WeightBlob conv2_result( 1 , 50 , 8 , 8 );
    conv2_gemm( pooling1 , kernel2 , bias2 , conv2_result );

    // pooling 2
    WeightBlob pooling2( 1 , 50 , 4 , 4 );
    max_pooling( conv2_result , pooling2 );

    // transfer the shape
    MatrixBlob from_pooling2( 1 , 800 );
    transfer2matrix( pooling2 , from_pooling2 );

    // inner product layer 1
    MatrixBlob inner1_result( 1 , 500 );
    MatrixBlob inner_weight1( net.layer(5).blobs(0) );
    BiasBlob bias3( net.layer(5).blobs(1) );
    inner_product( from_pooling2 , inner_weight1 , bias3 , inner1_result );

    //relu layer
    relu( inner1_result );

    // innner product layer2
    MatrixBlob inner2_result( 1 , 10 );
    MatrixBlob inner_weight2( net.layer(7).blobs(0) );
    BiasBlob bias4( net.layer(7).blobs(1) );
    inner_product( inner1_result , inner_weight2 , bias4 , inner2_result );

    for ( int i = 0 ; i < 10 ; i++ )
        score[i] = inner2_result.getValue( 0 , i );
}
