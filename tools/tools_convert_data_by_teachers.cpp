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
#include <io.h>
#include "tools_classifier.hpp"
#include "Binarizator\adaptive_threshold.hpp"
#include <direct.h>

using namespace std;
using namespace caffe;
using boost::scoped_ptr;
//float average_rho(vector<float> rho, float prec);

template<typename T>
T average_rho(vector<T> rho, T prec)
{
	vector< pair<T, int> > rho_counter;
	bool FOUND;
	for (int i = 0; i < rho.size(); i++)
	{
		FOUND = false;
		for (int j = 0; j < rho_counter.size(); j++)
		{
			if (fabs(rho[i] - rho_counter[j].first) < prec)
			{
				FOUND = true;
				rho_counter[j].second++;
				break;
			}
		}
		if (!FOUND)
		{
			rho_counter.push_back(make_pair(rho[i], 1));
		}
	}

	int counter_index = 0;
	int counter_num = rho_counter[0].second;

	for (int i = 0; i < rho_counter.size(); i++)
	{
		if (rho_counter[i].second > counter_num)
		{
			counter_num = rho_counter[i].second;
			counter_index = i;
		}
	}
	return rho_counter[counter_index].first;
}

template
float average_rho(vector<float> rho, float prec);
template
int average_rho(vector<int> rho, int prec);

void change_pic_names(const string & path, const string & target_path);
void rotate_pics(const string & target_path, const string & target_path2);
void make_grid(const string & target_path2, const string & target_path3);

int main( void )
{
	// 11111 change the names, do some boring stuff

	string path("C:\\Users\\JohnHush\\Desktop\\collected_data - GETTING-RID-OF-ONES-USING-BLACK-PENS\\");
	string target_path("C:\\Users\\JohnHush\\Desktop\\collected_data-PROCESSED\\");
	string target_path2("C:\\Users\\JohnHush\\Desktop\\collected_data-PROCESSED2\\");
	string target_path3("C:\\Users\\JohnHush\\Desktop\\collected_data-PROCESSED3\\");

	//change_pic_names( path , target_path );

	//rotate_pics( target_path , target_path2 );

	make_grid( target_path2, target_path3 );

	// 22222 rotate the image,

	//AdaThre * BINTOR = new AdaThre(201, 20);
	//target = target_path + "*.*";
	//handle = _findfirst(target.c_str(), &fileinfo);
	//if (handle == -1) return -1;

	//count = 0;

	//IplConvKernel * kernel = cvCreateStructuringElementEx(5, 5, 2, 2, CV_SHAPE_RECT);

	//do
	//{
	//	if (strcmp(fileinfo.name, ".") == 0 || strcmp(fileinfo.name, "..") == 0)
	//		continue;

	//	IplImage * imgSrc = cvLoadImage((target_path + fileinfo.name).c_str(), CV_LOAD_IMAGE_COLOR);
	//	IplImage * imgBin = cvCreateImage(cvSize(imgSrc->width, imgSrc->height), 8, 1);
	//	IplImage * imgHSV = cvCreateImage(cvSize(imgSrc->width, imgSrc->height), 8, 3);
	//	IplImage * imgRed = cvCreateImage(cvSize(imgSrc->width, imgSrc->height), 8, 1);
	//	IplImage * imgBla = cvCreateImage(cvSize(imgSrc->width, imgSrc->height), 8, 1);

	//	cvSetZero(imgBin);
	//	cvSetZero(imgRed);
	//	cvSetZero(imgHSV);
	//	cvSetZero(imgBla);

	//	BINTOR->binarizate(imgSrc, imgBin);
	//	cvCvtColor( imgSrc , imgHSV , CV_BGR2HSV );
	//	
	//	for ( int irow = 0 ; irow < imgBin->height ; ++ irow )
	//	for (int icol = 0; icol < imgBin->width; ++icol)
	//	{
	//		int HSV_H = cvGet2D( imgHSV , irow , icol ).val[0];
	//		if (cvGetReal2D(imgBin, irow, icol) != 255)
	//		{
	//			if (cvGet2D(imgHSV, irow, icol).val[1] > 50 && (HSV_H < 20 || HSV_H > 160))
	//				cvSetReal2D(imgRed, irow, icol, 255);
	//			else
	//				cvSetReal2D(imgBla, irow, icol, 255);
	//		}
	//	}

	//	IplImage * imgSrcClone = cvCloneImage(imgSrc);

	//	//cvErode(imgBla, imgBla, kernel);
	//	//cvDilate(imgBla, imgBla, kernel);

	//	CvMemStorage * storage = cvCreateMemStorage();
	//	CvSeq * contours;
	//	IplImage * imgRedClone = cvCloneImage(imgRed);
	//	cvFindContours(imgRed, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	//	vector< pair<CvRect, double> > contour_rect_area_pair;
	//	CvContour * contourGetter = (CvContour *)contours;

	//	do
	//	{
	//		pair<CvRect, double> tmp = make_pair( contourGetter->rect , fabs( cvContourArea(contourGetter)) );
	//		contour_rect_area_pair.push_back(tmp);
	//		contourGetter = (CvContour *)contourGetter->h_next;
	//	} while (contourGetter != 0);

	//	sort( contour_rect_area_pair.begin() , contour_rect_area_pair.end() , sort_area );

	//	for (int index = 0; index < contour_rect_area_pair.size() ; ++index)
	//	{
	//		CvRect & rect = contour_rect_area_pair[index].first;
	//		CvPoint point1 = cvPoint( rect.x , rect.y );
	//		CvPoint point2 = cvPoint( rect.x + rect.width , rect.y + rect.height );
	//		//cvRectangle(imgSrc, point1 , point2 , CV_RGB(0 , 255 , 0 ), 3);
	//	}

	//	CvSeq * lines = 0;
	//	//lines = cvHoughLines2(imgBla, storage, CV_HOUGH_PROBABILISTIC, 3, CV_PI / 180, 400, 60, 3 );
	//	lines = cvHoughLines2(imgBla, storage, CV_HOUGH_STANDARD , 1 , CV_PI / 180, 400, 60, 3);
	//	
	//	vector<float> theta_vec;

	//	for (int i = 0; i < lines->total; ++i)
	//	{
	//		float * line = (float*)cvGetSeqElem(lines, i);
	//		float rho = line[0];
	//		float theta = line[1];

	//		if (theta > 1.31 && theta < 1.84)
	//			theta_vec.push_back( theta );
	//	}

	//	float ave = average_rho(theta_vec, 0.01 );

	//	cout << "count=" <<count << "  average angle = " << ave << endl;

	//	CvPoint2D32f center;
	//	center.x = float( imgSrc->width/2. + 0.5 );
	//	center.y = float( imgSrc->height/2. + 0.5 );

	//	float degree = -90. + ave * 180. / CV_PI;
	//	float m[6];
	//	CvMat M = cvMat( 2, 3, CV_32F , m );
	//	cv2DRotationMatrix( center , degree , 1 , &M );

	//	IplImage * imgSrcRot = cvCreateImage(cvSize(imgSrc->width, imgSrc->height), 8, 3 );
	//	cvWarpAffine( imgSrc , imgSrcRot , &M , CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, cvScalarAll(255) );

	//	for (int i = 0; i < lines->total; ++i)
	//	{
	//		CvPoint * line = (CvPoint*)cvGetSeqElem(lines, i);
	//		cvLine(imgSrcClone, line[0], line[1], CV_RGB(0, 255, 0), 1, CV_AA);
	//	}


	//	cvSaveImage( (target_path2 + fileinfo.name ).c_str() , imgSrcRot );
	//	//showImage(imgBla, 0.4 , string("showImage") , 0);
	//	count++;

	//	cvReleaseImage(&imgSrc);
	//	cvReleaseImage(&imgBin);
	//	cvReleaseImage(&imgHSV);
	//	cvReleaseImage(&imgRed);
	//	cvReleaseImage(&imgBla);
	//	cvReleaseMemStorage(&storage);
	//	cvReleaseImage(&imgRedClone);
	//	cvReleaseImage(&imgSrcClone);
	//	cvReleaseImage(&imgSrcRot);

	//} while (_findnext(handle, &fileinfo) == 0);


	//33333 find the 1,2,3,4,etc characters in the front paper, BOLD ONES
	


	// hahahahhahahahahaahahahahahahahahahahha
	return 1;
}

void make_grid( const string & target_path2 , const string & target_path3 )
{
	string target_path4("C:\\Users\\JohnHush\\Desktop\\collected_data-PROCESSED4\\");
	string target_path5("C:\\Users\\JohnHush\\Desktop\\collected_data-PROCESSED5\\");
	for (int i = 0; i < 10; i++)
	{
		mkdir((target_path4 + std::to_string(i)).c_str());
		mkdir((target_path5 + std::to_string(i)).c_str());
	}

	AdaThre * BINTOR = new AdaThre(201, 20);
	string target = target_path2 + "*.*";

	struct _finddata_t fileinfo;
	long handle = _findfirst(target.c_str(), &fileinfo);
	if (handle == -1) return;

	IplConvKernel * kernel = cvCreateStructuringElementEx(7, 7, 3, 3, CV_SHAPE_RECT);
	IplConvKernel * kernel2 = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_RECT);

	
	int TTT = 0;

	do
	{
		cout << "TTT= " << TTT << endl;
		CvMemStorage * storage = cvCreateMemStorage();
		TTT++;
		if (strcmp(fileinfo.name, ".") == 0 || strcmp(fileinfo.name, "..") == 0)
			continue;

		IplImage * imgSrc = cvLoadImage((target_path2 + fileinfo.name).c_str(), CV_LOAD_IMAGE_COLOR);
		IplImage * imgBin = cvCreateImage(cvSize(imgSrc->width, imgSrc->height), 8, 1);
		IplImage * imgHSV = cvCreateImage(cvSize(imgSrc->width, imgSrc->height), 8, 3);
		IplImage * imgRed = cvCreateImage(cvSize(imgSrc->width, imgSrc->height), 8, 1);
		IplImage * imgBla = cvCreateImage(cvSize(imgSrc->width, imgSrc->height), 8, 1);

		cvSetZero(imgBin);
		cvSetZero(imgRed);
		cvSetZero(imgHSV);
		cvSetZero(imgBla);

		BINTOR->binarizate(imgSrc, imgBin);
		cvCvtColor(imgSrc, imgHSV, CV_BGR2HSV);

		for (int irow = 0; irow < imgBin->height; ++irow)
			for (int icol = 0; icol < imgBin->width; ++icol)
			{
				int HSV_H = cvGet2D(imgHSV, irow, icol).val[0];
				if (cvGetReal2D(imgBin, irow, icol) != 255)
				{
					if (cvGet2D(imgHSV, irow, icol).val[1] > 50 && (HSV_H < 20 || HSV_H > 160))
						cvSetReal2D(imgRed, irow, icol, 255);
					else
						cvSetReal2D(imgBla, irow, icol, 255);
				}
			}
		IplImage * imgBlaClone = cvCloneImage(imgBla);	

		

		CvSeq * lines = 0;
		lines = cvHoughLines2(imgBla, storage, CV_HOUGH_PROBABILISTIC, 4, CV_PI / 180, 40, 60, 5 );
		//	lines = cvHoughLines2(imgBla, storage, CV_HOUGH_STANDARD , 1 , CV_PI / 180, 400, 60, 3);
		//	
		vector<int> horiz_vec;
		vector<int> verti_vec;

		for (int i = 0; i < lines->total; ++i)
			{
				CvPoint * line = (CvPoint *)cvGetSeqElem(lines, i);

				//cvLine( imgSrc ,  line[0] , line[1] , CV_RGB(0,255,0) , 3 );

				CvPoint & pt1 = line[0];
				CvPoint & pt2 = line[1];
				if (abs(pt1.x - pt2.x) > abs(pt1.y - pt2.y))
					horiz_vec.push_back(pt1.y);
				else
					verti_vec.push_back( (pt1.y+pt2.y)/2 );
			}
			std::sort( horiz_vec.begin() , horiz_vec.end() );
			int find_hori = imgSrc->height/10;

			for (int i = 0; i < horiz_vec.size(); ++i)
			{
				int red_counter = 0;
				if (horiz_vec[i] > find_hori)
				{
					find_hori = horiz_vec[i];

					for (int icol = 0; icol < imgRed->width; ++icol)
						if (cvGetReal2D(imgRed, find_hori+ imgSrc->height / 40, icol) == 255)
							red_counter++;
					if (red_counter > 3)
						break;
					else
						continue;
				}
			}
			

			// find the gap between the grids...

			vector<int>::iterator iter;
			for (iter = horiz_vec.begin(); iter != horiz_vec.end(); )
			{
				if (*iter < find_hori || *iter > imgRed->height * 0.75)
				{
					iter = horiz_vec.erase(iter);
					//iter--;
				}
				else
					iter++;
			}
		
			std::sort( horiz_vec.begin() , horiz_vec.end() );

			vector<int> GAP_GATHER;
			int GAP = imgRed->width/100;

			for (int i = 0; i < horiz_vec.size(); i++)
			{
				int  current_x = horiz_vec[i];
				for (int j = i; j < horiz_vec.size(); j++)
				{
					if (abs(current_x - horiz_vec[j]) < GAP)
						continue;
					else
					{
						GAP_GATHER.push_back(abs(current_x - horiz_vec[j]));
						break;
					}
				}
			}

			// use k means to cluster this dataset;
			int KMEANS_ITER = 0;
			int ave1, ave2;
			vector<int> group1, group2;
			do
			{
				group1.clear();
				group2.clear();
				if (KMEANS_ITER == 0)
				{
					ave1 = GAP_GATHER[0];
					ave2 = GAP_GATHER[GAP_GATHER.size()-1];
				}
				for (int i = 0; i < GAP_GATHER.size(); i++)
				{
					if (abs(GAP_GATHER[i] - ave1) < abs(GAP_GATHER[i] - ave2))
						group1.push_back(GAP_GATHER[i]);
					else
						group2.push_back(GAP_GATHER[i]);
				}

				ave1 = average_rho( group1 , GAP );
				ave2 = average_rho(group2, GAP);

				KMEANS_ITER++;
			} while (KMEANS_ITER < 5);

			int BIG_GAP = std::max( ave1 , ave2 );
			int SMA_GAP = std::min( ave1 , ave2 );
			int start_ = find_hori + BIG_GAP/2 ;

			/*for (int i = 0; i < 10; i++)
			{
				cvLine( imgSrc , cvPoint(0, start_  ) , cvPoint(imgSrc->width, start_) , CV_RGB(0,0,255), 5);
				start_ += BIG_GAP + SMA_GAP;
			}*/

			int rect_width = imgBlaClone->width;
			int rect_height = imgSrc->height / 30;
			int rect_x = 0;
			int rect_y = find_hori - imgSrc->height / 20;
			cvSetImageROI( imgBlaClone, cvRect( rect_x , rect_y , rect_width , rect_height ) );
			cvSetImageROI(imgSrc, cvRect(rect_x, rect_y, rect_width, rect_height));
			IplImage* imgBlaCopy = cvCreateImage( cvSize(rect_width , rect_height ), 8 , 1);
			
			
			cvDilate(imgBlaClone, imgBlaClone, kernel);
			cvErode(imgBlaClone, imgBlaClone, kernel);

			cvErode(imgBlaClone, imgBlaClone, kernel2);
			cvDilate(imgBlaClone, imgBlaClone, kernel2);
			cvCopy(imgBlaClone, imgBlaCopy);

			int red_index_left = 0;
			int red_index_righ = imgRed->width-1;
			
			for (int icol = 0; icol < imgRed->width; ++icol)
			{
				int sum = 0;
				for (int irow = 0; irow < imgRed->height; ++irow)
					if (cvGetReal2D(imgRed, irow, icol) == 255)
						sum++;
				if (sum > 20)
				{
					red_index_left = icol;
					break;
				}
			}

			for (int icol = imgRed->width -1; icol >=0 ; --icol)
			{
				int sum = 0;
				for (int irow = 0; irow < imgRed->height; ++irow)
					if (cvGetReal2D(imgRed, irow, icol) == 255)
						sum++;
				if (sum > 20)
				{
					red_index_righ = icol;
					break;
				}
			}
			//cvLine(imgRed, cvPoint(red_index_righ + imgBlaCopy->width / 60, 0), cvPoint(red_index_righ + imgBlaCopy->width / 60, imgRed->height), cvScalar(255), 3);
			//cvLine(imgRed, cvPoint(red_index_left - imgBlaCopy->width / 60, 0), cvPoint(red_index_left - imgBlaCopy->width / 60, imgRed->height), cvScalar(255), 3);

			for (int irow = 0; irow < imgBlaCopy->height; ++irow)
				for (int icol = 0; icol < imgBlaCopy->width; ++icol)
					if (icol < red_index_left - imgBlaCopy->width / 60 || icol >  red_index_righ + imgBlaCopy->width / 60 )
						cvSetReal2D(imgBlaCopy, irow , icol , 0 );

			CvSeq * contours = nullptr ;
			//	IplImage * imgRedClone = cvCloneImage(imgRed);
			cvFindContours(imgBlaCopy, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
			vector< pair<CvRect, double> > contour_rect_area_pair;
			CvContour * contourGetter = (CvContour *)contours;

			do
			{
			pair<CvRect, double> tmp = make_pair( contourGetter->rect , fabs( cvContourArea(contourGetter)) );
			contour_rect_area_pair.push_back(tmp);
					contourGetter = (CvContour *)contourGetter->h_next;
			} while (contourGetter != 0);

			sort( contour_rect_area_pair.begin() , contour_rect_area_pair.end() , sort_area );

			vector<int> center_x;
			for (int i = 0; i < 10; i++)
				center_x.push_back(contour_rect_area_pair[i].first.x + contour_rect_area_pair[i].first.width/2 );

			std::sort( center_x.begin() , center_x.end() );
			int left_most = imgRed->width - 1;
			int righ_most = 0;

			for (int i = 0; i < 10; i++)
			{
				if (center_x[i] > righ_most)
					righ_most = center_x[i];
				if (center_x[i] < left_most)
					left_most = center_x[i];
			}

			int gap = (righ_most - left_most) / 9;


			cvResetImageROI(imgSrc);

			// center_x ,,,find_hori , BIG_GAP , SMA_GAP...
			// after we got these parameters, we can reestimate the pics.
			IplImage* imgRedClone = cvCloneImage(imgRed);
			CvSeq * red_contours = NULL ;
			cvFindContours(imgRedClone, storage, &red_contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

			vector< pair<CvRect, double> > red_contour_area_pair;
			CvContour * red_contourGetter = (CvContour *)red_contours;

			do
			{
				pair<CvRect, double> tmp = make_pair(red_contourGetter->rect, fabs(cvContourArea(red_contourGetter)));
				red_contour_area_pair.push_back(tmp);
				red_contourGetter = (CvContour *)red_contourGetter->h_next;
			} while (red_contourGetter != 0);



			vector< vector< pair<CvRect, double> > > arranged_rect(100);

			int AVE_X_GAP = 0;
			int AVE_Y_GAP = BIG_GAP + SMA_GAP;

			for (int index = 1; index < center_x.size(); ++index)
			{
				AVE_X_GAP += center_x[index] - center_x[index-1];
			}
			AVE_X_GAP /= (center_x.size()-1);


			vector<int> center_y(10);
			for (int i = 0; i < 10; i++)
				center_y[i] = find_hori + BIG_GAP/2 + i * (BIG_GAP+ SMA_GAP);
			for (int index = 0; index < red_contour_area_pair.size(); ++index)
			{
				pair<CvRect, double> & pairs_tmp = red_contour_area_pair[index];
				CvRect & rect = red_contour_area_pair[index].first;
				CvPoint center = cvPoint( rect.x + rect.width/2 , rect.y + rect.height/2 );
				if (center.x < center_x[0] - AVE_X_GAP / 2 || center.x > center_x[9] + AVE_X_GAP / 2 ||
					center.y < center_y[0] - AVE_Y_GAP / 2 || center.y > center_y[9] + AVE_Y_GAP / 2)
					continue;

				int X_INDEX = 0;
				int Y_INDEX = 0;

				for ( int i = 0 ; i < 10 ; i ++ )
					if (abs(center.x - center_x[i]) <= AVE_X_GAP / 2)
					{
						X_INDEX = i;
						break;
					}
				for (int i = 0; i < 10; i++)
					if (abs(center.y - center_y[i]) <= AVE_Y_GAP / 2)
					{
						Y_INDEX = i;
						break;
					}

				arranged_rect[Y_INDEX * 10 + X_INDEX].push_back(pairs_tmp);
			}

			for (int i = 0; i < 100; i++)
				sort( arranged_rect[i].begin() , arranged_rect[i].end() , sort_area );

			vector<CvRect> mergingBox(100);

			for (int index = 0; index < 100; index++)
			{
				if (arranged_rect[index].size() == 0)
				{
					mergingBox[index] = cvRect(-1,-1,-1,-1);
					continue;
				}
				int start_area = arranged_rect[index][0].second;
				mergingBox[index] = arranged_rect[index][0].first;
				for (int jndex = 1; jndex < arranged_rect[index].size(); ++jndex)
				{
					CvRect & next_rect = arranged_rect[index][jndex].first;
					//float ratio = std::max(next_rect.height,next_rect.width) / std::min(next_rect.height, next_rect.width);
					if (arranged_rect[index][jndex].second > start_area * 0.05 )
					{
						start_area += arranged_rect[index][jndex].second;
						merging_box(mergingBox[index], arranged_rect[index][jndex].first);
					}
					else
						break;
				}

				CvPoint pt1 = cvPoint(mergingBox[index].x , mergingBox[index].y );
				CvPoint pt2 = cvPoint(mergingBox[index].x + mergingBox[index].width, mergingBox[index].y+ mergingBox[index].height);
				//cvRectangle(imgSrc, pt1, pt2, CV_RGB(0, 0, 255), 2);
			}

			for (int index = 0; index < 100; index++)
			{
				if (mergingBox[index].x < 0)
					continue;
				int dire_index = index % 10;
				string name_file = target_path4 + to_string(dire_index) + "//" + to_string(index) + fileinfo.name;
				string name_file2 = target_path5 + to_string(dire_index) + "//" + to_string(index) + fileinfo.name;

				CvRect & BOUNDING_BOX = mergingBox[index];

				cvSetImageROI(imgRed, BOUNDING_BOX);
				IplImage * imgRedROI = cvCloneImage(imgRed);

				int box_width = BOUNDING_BOX.width;
				int box_heigh = BOUNDING_BOX.height;

				float scale = 200 / float(box_width>box_heigh ? box_width : box_heigh);

				int tmp_width = int(280 / scale);
				int tmp_heigh = int(280 / scale);

				IplImage * imgTMP = cvCreateImage(cvSize(tmp_width, tmp_heigh), 8, 1);
				cvSetZero(imgTMP);

				int xGap = (tmp_width - box_width) / 2;
				int yGap = (tmp_heigh - box_heigh) / 2;

				for (int ir = 0; ir < box_heigh; ++ir)
					for (int ic = 0; ic < box_width; ++ic)
						cvSetReal2D(imgTMP, ir + yGap, ic + xGap, cvGetReal2D(imgRedROI, ir, ic));

				IplImage * imgTMP2 = cvCreateImage(cvSize(280, 280), 8, 1);
				IplImage * imgRst = cvCreateImage(cvSize(280, 280), 8, 1);
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




				cvSetImageROI( imgSrc , BOUNDING_BOX );

				cvSaveImage( name_file.c_str(), imgSrc);
				cvSaveImage(name_file2.c_str(), imgRst);

				cvResetImageROI(imgSrc);
				cvResetImageROI(imgRed);

				cvReleaseImage(&imgRedROI);
				cvReleaseImage(&imgTMP);
				cvReleaseImage(&imgTMP2);
				cvReleaseImage(&imgRst);

			}
		
			







			/*for (int i = 0; i < 10; i++)
			{
				CvRect & rect = contour_rect_area_pair[i].first;
				CvPoint pt1 = cvPoint( rect.x , rect.y );
				CvPoint pt2 = cvPoint( rect.x + rect.width , rect.y + rect.height );
				cvRectangle(imgSrc, pt1, pt2, CV_RGB(0,255,0), 3);
			}*/

		//for ( int i = 0 ; i < center_x.size() ; i++ )
		//	cvLine(imgSrc, cvPoint( center_x[i], 0 ), cvPoint(center_x[i], imgSrc->height ), CV_RGB(0, 0, 255), 5);
		//cvLine(imgSrc, cvPoint(0 , find_hori + gap * 0.4  ), cvPoint(imgSrc->width, find_hori + gap*0.4 ), CV_RGB(0, 0, 255), 3);
		
		//cvLine(imgSrc, cvPoint(0, find_hori+ imgSrc->height / 40), cvPoint(imgSrc->width, find_hori+imgSrc->height/40), 
		//	CV_RGB(0, 0, 255), 3);
		cvSaveImage((target_path3 + fileinfo.name).c_str(), imgSrc);
		
		cvReleaseImage(&imgSrc);
		cvReleaseImage(&imgBin);
		cvReleaseImage(&imgHSV);
		cvReleaseImage(&imgRed);
		cvReleaseImage(&imgBla);
		cvReleaseImage(&imgBlaClone);
		cvReleaseImage( &imgBlaCopy);
		cvReleaseImage(&imgRedClone);
		cvReleaseMemStorage(&storage);

	} while (_findnext(handle, &fileinfo) == 0);

	

}

void rotate_pics( const string & target_path , const string & target_path2 )
{
	AdaThre * BINTOR = new AdaThre(201, 20);
	string target = target_path + "*.*";
	struct _finddata_t fileinfo;
	long handle = _findfirst(target.c_str(), &fileinfo);
	if (handle == -1) return;

	int count = 0;

	do
	{
		if (strcmp(fileinfo.name, ".") == 0 || strcmp(fileinfo.name, "..") == 0)
			continue;

		IplImage * imgSrc = cvLoadImage((target_path + fileinfo.name).c_str(), CV_LOAD_IMAGE_COLOR);
		IplImage * imgBin = cvCreateImage(cvSize(imgSrc->width, imgSrc->height), 8, 1);
		IplImage * imgHSV = cvCreateImage(cvSize(imgSrc->width, imgSrc->height), 8, 3);
		IplImage * imgRed = cvCreateImage(cvSize(imgSrc->width, imgSrc->height), 8, 1);
		IplImage * imgBla = cvCreateImage(cvSize(imgSrc->width, imgSrc->height), 8, 1);

		cvSetZero(imgBin);
		cvSetZero(imgRed);
		cvSetZero(imgHSV);
		cvSetZero(imgBla);

		BINTOR->binarizate(imgSrc, imgBin);
		cvCvtColor(imgSrc, imgHSV, CV_BGR2HSV);

		for (int irow = 0; irow < imgBin->height; ++irow)
			for (int icol = 0; icol < imgBin->width; ++icol)
			{
				int HSV_H = cvGet2D(imgHSV, irow, icol).val[0];
				if (cvGetReal2D(imgBin, irow, icol) != 255)
				{
					if (cvGet2D(imgHSV, irow, icol).val[1] > 50 && (HSV_H < 20 || HSV_H > 160))
						cvSetReal2D(imgRed, irow, icol, 255);
					else
						cvSetReal2D(imgBla, irow, icol, 255);
				}
			}

		IplImage * imgSrcClone = cvCloneImage(imgSrc);

		CvMemStorage * storage = cvCreateMemStorage();
		
		CvSeq * lines = 0;
		//lines = cvHoughLines2(imgBla, storage, CV_HOUGH_PROBABILISTIC, 3, CV_PI / 180, 400, 60, 3 );
		lines = cvHoughLines2(imgBla, storage, CV_HOUGH_STANDARD, 1, CV_PI / 180, 400, 60, 3);

		vector<float> theta_vec;
		theta_vec.clear();

		for (int i = 0; i < lines->total; ++i)
		{
			float * line = (float*)cvGetSeqElem(lines, i);
			float rho = line[0];
			float theta = line[1];

			if (theta > 1.31 && theta < 1.84)
				theta_vec.push_back(theta);
		}
		float ave=0;

		for (int i = 0; i < theta_vec.size(); i++)
			ave += theta_vec[i] / theta_vec.size();

		//float ave = average_rho(theta_vec, 0.001);

		CvPoint2D32f center;
		center.x = float(imgSrc->width / 2. + 0.5);
		center.y = float(imgSrc->height / 2. + 0.5);

		float degree = -90. + ave * 180. / CV_PI;

		float m[6];
		CvMat M = cvMat(2, 3, CV_32F, m);
		cv2DRotationMatrix(center, degree, 1, &M);

		IplImage * imgSrcRot = cvCreateImage(cvSize(imgSrc->width, imgSrc->height), 8, 3);
		
		cvWarpAffine(imgSrc, imgSrcRot, &M, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(255));

		cvSaveImage((target_path2 + fileinfo.name).c_str(), imgSrcRot);
		//showImage(imgBla, 0.4 , string("showImage") , 0);
		count++;

		cvReleaseImage(&imgSrc);
		cvReleaseImage(&imgBin);
		cvReleaseImage(&imgHSV);
		cvReleaseImage(&imgRed);
		cvReleaseImage(&imgBla);
		cvReleaseMemStorage(&storage);
		cvReleaseImage(&imgSrcClone);
		cvReleaseImage(&imgSrcRot);

	} while (_findnext(handle, &fileinfo) == 0);

	delete BINTOR;
}

void change_pic_names( const string & path , const string & target_path )
{
	long handle;
	struct _finddata_t fileinfo;
	string target = path + "*.*";
	char key_num[10];

	handle = _findfirst(target.c_str(), &fileinfo);
	if (handle == -1) return;
	int count = 0;
	do
	{
		if (strcmp(fileinfo.name, ".") == 0 || strcmp(fileinfo.name, "..") == 0)
			continue;

		snprintf(key_num, 10, "%08d", count);

		CopyFile((path + fileinfo.name).c_str(), (target_path + key_num + ".jpg").c_str(), FALSE);
		count++;
	} while (_findnext(handle, &fileinfo) == 0);
}




