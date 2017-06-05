#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include <string>
#include "util.hpp"
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <memory>
#include "boost/scoped_ptr.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"

using namespace std;
using namespace caffe;
using boost::scoped_ptr;

int main( void )
{
vector< pair<char* , int> > MyData;
for ( int index_img = 1 ; index_img < 23 ; index_img++ )
{
	ostringstream stream;
	stream << index_img;
	string index_num = stream.str();
	if ( index_img < 10 )
		index_num = string("0") + index_num;
#ifdef UNIX
	string IMG_NAME = "/home/pitaloveu/Caffe2Windows/finetune_data_withoutBOX/Page00" + index_num + ".jpg";
#endif
#ifdef _WINDOWS
	string IMG_NAME = "D:\\MyProjects\\orion-eye\\finetune_data_withoutBOX\\Page00" + index_num + ".jpg";
#endif
//	string IMG_NAME( "/home/pitaloveu/Caffe2Windows/finetune_data_withoutBOX\\Page0015.jpg" );
	IplImage * imgSrc = cvLoadImage( IMG_NAME.c_str() , CV_LOAD_IMAGE_GRAYSCALE );

	int width = imgSrc->width;
	int heigh = imgSrc->height;

	IplImage * imgBin = cvCreateImage( cvSize( width , heigh ) , 8 , 1 );
	IplImage * imgGray = cvCreateImage( cvSize( width , heigh ) , 8 , 1 );

	cvThreshold( imgSrc , imgBin , 230 , 255 , CV_THRESH_BINARY_INV );

	cvSetZero( imgGray );
	for ( int irow = 0 ; irow < heigh ; ++irow )
	for ( int icol = 0 ; icol < width ; ++icol )
		if ( cvGetReal2D( imgBin , irow , icol ) != 0 )
			cvSetReal2D( imgGray , irow , icol , cvGetReal2D( imgSrc , irow , icol ) );

	IplImage * img4Contour = cvCloneImage( imgBin );

	// find all the contours in the image
	CvMemStorage * storage = cvCreateMemStorage();
	CvSeq * contours;
	cvFindContours( img4Contour , storage , &contours , sizeof(CvContour) , CV_RETR_EXTERNAL , CV_CHAIN_APPROX_SIMPLE );
	cvSetZero( img4Contour );

	CvContour * contourGetter = (CvContour *)contours;
	vector< pair<CvRect , double> > contour_rect_area_pair;
	vector< pair<CvRect , double> > contour_rect_area_pair_whole;

	do
	{
		CvRect & rect = contourGetter->rect;
		
		CvPoint pt1 = cvPoint( rect.x , rect.y );
		CvPoint pt2 = cvPoint( rect.x + rect.width , rect.y + rect.height );

//		cvRectangle( imgBin , pt1 , pt2 , cvScalar(255) );

		pair<CvRect , double> tmp = make_pair( contourGetter->rect , fabs(cvContourArea( contourGetter )) );
		contour_rect_area_pair.push_back( tmp );
		contour_rect_area_pair_whole.push_back( tmp );

		contourGetter = ( CvContour * ) contourGetter->h_next;
	}
	while ( contourGetter != 0 );
	sort( contour_rect_area_pair.begin() , contour_rect_area_pair.end() , sort_area );
	contour_rect_area_pair.resize( 100 );

	vector<int> xcor(10,0);
	vector<int> ycor(10,0);

	sort( contour_rect_area_pair.begin() , contour_rect_area_pair.end() , sort_rect_area_pair_x );
	xcor[0] = contour_rect_area_pair[99].first.x + contour_rect_area_pair[99].first.width/2;
	xcor[9] = contour_rect_area_pair[0].first.x + contour_rect_area_pair[0].first.width/2;
	sort( contour_rect_area_pair.begin() , contour_rect_area_pair.end() , sort_rect_area_pair_y );
	ycor[0] = contour_rect_area_pair[99].first.y + contour_rect_area_pair[99].first.height/2;
	ycor[9] = contour_rect_area_pair[0].first.y + contour_rect_area_pair[0].first.height/2;

	int xgap = ( xcor[9] - xcor[0] )/9;
	int ygap = ( ycor[9] - ycor[0] )/9;

	for ( int i = 1 ; i < 9 ; i++ )
	{
		xcor[i] = xcor[i-1] + xgap;
		ycor[i] = ycor[i-1] + ygap;
	}

	for ( int irow = 0 ; irow < 10 ; irow ++ )
	for ( int icol = 0 ; icol < 10 ; icol ++ )
	{
		int XCOR = xcor[icol];
		int YCOR = ycor[irow];
		vector< pair<CvRect , double> > Contours41Cha;
		for ( int icontour = 0 ; icontour < contour_rect_area_pair_whole.size() ; ++ icontour )
		{
			CvRect & tmp = contour_rect_area_pair_whole[icontour].first;
			CvPoint center = cvPoint( tmp.x + tmp.width/2 , tmp.y + tmp.height/2 );
			if ( abs(center.x - XCOR) >= xgap/2 || abs(center.y - YCOR) >= ygap/2 )
				continue;

			Contours41Cha.push_back( contour_rect_area_pair_whole[icontour] );
		}
		sort( Contours41Cha.begin() , Contours41Cha.end() , sort_area );
		//find out the fix box;
		double area = 0.;
		for ( int i = 0 ; i < Contours41Cha.size() ; i++ )
			area += Contours41Cha[i].second;

		CvRect BBOX = Contours41Cha[0].first;
		double adding_area = Contours41Cha[0].second;
		int index = 1;
		while( adding_area < 0.92 * area )
		{
			merging_box( BBOX , Contours41Cha[index].first );
			adding_area += Contours41Cha[index].second;
			index++;
		}
		cvSetImageROI( imgGray , BBOX );
//		showImage( imgBin , 0.2 , "test" , 500 );

		int box_width = BBOX.width;
		int box_heigh = BBOX.height;

		float scale = 20 / float(box_width>box_heigh?box_width:box_heigh);

		int tmp_width = int(28 / scale);
		int tmp_heigh = int(28 / scale);

		IplImage * imgTMP = cvCreateImage( cvSize( tmp_width , tmp_heigh ) , 8 , 1 );
		cvSetZero( imgTMP );

		int xGap = ( tmp_width - box_width ) / 2;
		int yGap = ( tmp_heigh - box_heigh ) / 2;

		for ( int ir = 0 ; ir < box_heigh ; ++ ir )
			for ( int ic = 0 ; ic < box_width ; ++ ic )
				cvSetReal2D( imgTMP , ir + yGap , ic + xGap , cvGetReal2D( imgGray , ir , ic ) );

		IplImage * imgTMP2 = cvCreateImage( cvSize( 28 , 28 ) , 8 , 1 );
		cvSetZero( imgTMP2 );

		cvResize( imgTMP , imgTMP2 , CV_INTER_AREA );

		int xHeart_resize = 0;
		int yHeart_resize = 0;
		int count_resize = 0;

		for ( int ir = 0 ; ir < 28 ; ++ ir )
			for ( int ic = 0 ; ic < 28 ; ++ ic )
			{
				if ( cvGetReal2D( imgTMP2 , ir , ic ) != 0 )
				{
					xHeart_resize += ic;
					yHeart_resize += ir;
					count_resize ++;
				}
			}
		xHeart_resize /= count_resize;
		yHeart_resize /= count_resize;

		IplImage * imgRst = cvCreateImage( cvSize( 28 , 28 ) , 8 , 1 );

		cvSetZero( imgRst );

		for ( int ir = 0 ; ir < 28 ; ++ ir )
            for ( int ic = 0 ; ic < 28 ; ++ ic )
            {
                int xcor = ic - 14 + xHeart_resize;
                int ycor = ir - 14 + yHeart_resize;
                if ( xcor >= 0 && xcor < 28 && ycor >= 0 && ycor < 28 )
                    cvSetReal2D( imgRst , ir , ic , cvGetReal2D( imgTMP2 , ycor , xcor ) );
            }
/*
		for ( int irow = 0 ; irow < 28 ; ++ irow )
            for ( int icol = 0 ; icol < 28 ; ++ icol )
                if ( cvGetReal2D( imgRst , irow , icol ) != 0 )
                    cvSetReal2D( imgRst , irow , icol , 255);
*/
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
		
//		showImage( imgRst , 10 , "seesee" , 500 );

		char * pixels = new char [ 28 * 28 ];
		
		for ( int ir = 0 ; ir < 28 ; ++ ir)
		{
			char * ptr = (char *) imgRst->imageData + ir * imgRst->widthStep;
			for ( int ic = 0 ; ic < 28 ; ++ ic )
			{
				pixels[ ir * 28 + ic ] = ptr[ic];
			}
		}
		MyData.push_back( make_pair( pixels , icol ) );

//		if ( index_img == 1 && icol == 3 && irow == 9 )
//		{
//		 for ( int ii = 0 ; ii < 28 * 28  ; ii ++ )
//			 cout << "ii = " << ii << "  value = " << int(((uchar *)pixels)[ii]) << endl;
//		}
		
		cvReleaseImage( &imgRst );
		cvReleaseImage( &imgTMP2 );
		cvReleaseImage( &imgTMP );

		cvResetImageROI( imgGray );
	}
	cvReleaseImage( &imgSrc );
	cvReleaseImage( &imgBin );
	cvReleaseImage( &imgGray );
	cvReleaseImage( &img4Contour );
	cvReleaseMemStorage( &storage );
//	showImage( imgBin , 0.3 , "test" , 0 );
}
random_shuffle( MyData.begin() , MyData.end() );

if ( false ){
/*
 * write leveldb format data
 * using the original API
 */
	leveldb::DB * db;
	leveldb::Options options;
	options.error_if_exists = true;
	options.create_if_missing = true;
	leveldb::WriteBatch * batch = new leveldb::WriteBatch() ;

	leveldb::Status status = leveldb::DB::Open( options , 
				"/home/pitaloveu/Caffe2Windows/finetune_data_withoutBOX/finetune_training_data_leveldb" , &db );

	Datum datum;
	datum.set_channels(1);
	datum.set_height(28);
	datum.set_width(28);
	string value;

	for ( int item_id = 0 ; item_id < 1840 ; item_id ++ )
	{
		datum.set_data( MyData[item_id].first , 28 * 28 );
		datum.set_label( MyData[item_id].second );
		string key_str = caffe::format_int(item_id, 8);
		datum.SerializeToString(&value);

		batch->Put( key_str , value );

	}
	db->Write( leveldb::WriteOptions() , batch );
	delete batch;
	delete db;
}
scoped_ptr<db::DB> db(db::GetDB(string("leveldb")));
#ifdef _WINDOWS
db->Open( "D:\\MyProjects\\orion-eye\\finetune_data_withoutBOX\\finetune_training_data_leveldb" , db::NEW);
#endif
#ifdef UNIX
db->Open( "/home/pitaloveu/Caffe2Windows/finetune_data_withoutBOX/finetune_training_data_leveldb" , db::NEW);
#endif
scoped_ptr<db::Transaction> txn(db->NewTransaction());

Datum datum;
datum.set_channels(1);
datum.set_height(28);
datum.set_width(28);
string value;

for ( int item_id = 0 ; item_id < 1840 ; item_id ++ )
{
datum.set_data( MyData[item_id].first , 28 * 28 );
datum.set_label( MyData[item_id].second );
string key_str = caffe::format_int(item_id, 8);
datum.SerializeToString(&value);
txn->Put(key_str, value);
}
txn->Commit();
db->Close();

#ifdef _WINDOWS
db->Open( "D:\\MyProjects\\orion-eye\\finetune_data_withoutBOX\\finetune_testing_data_leveldb" , db::NEW);
#endif
#ifdef UNIX
db->Open( "/home/pitaloveu/Caffe2Windows/finetune_data_withoutBOX/finetune_testing_data_leveldb" , db::NEW);
#endif
scoped_ptr<db::Transaction> txn2(db->NewTransaction());

Datum datum2;
datum2.set_channels(1);
datum2.set_height(28);
datum2.set_width(28);
string value2;

for ( int item_id = 1840 ; item_id < MyData.size() ; item_id ++ )
{
datum2.set_data( MyData[item_id].first , 28 * 28 );
datum2.set_label( MyData[item_id].second );
string key_str = caffe::format_int(item_id, 8);
datum2.SerializeToString(&value2);
txn2->Put(key_str, value2);
}
txn2->Commit();
db->Close();


//end
}
