#define NOMINMAX
#define NO_STRICT
#include "HandWritingDigitsRecognitionSystem.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include "util.hpp"
#include <io.h>
#include <stdio.h>
#include "util_caffe.hpp"
//#include "caffe.pb.h"

//int getAllImages(vector< pair<string, int> > & imgName, string path, int LABEL= -1);
int main( int argc , char * argv[] )
{
	initPredictor();
#ifdef _WINDOWS
	IplImage * imgSrc = cvLoadImage( "C:\\handwriting\\20170421\\before\\3__0.34__4564838.jpg" , CV_LOAD_IMAGE_COLOR );
#endif
#ifdef UNIX
	IplImage * imgSrc = cvLoadImage( "/home/pitaloveu/orion-eye/test_data/color_5.jpg" , CV_LOAD_IMAGE_COLOR );
#endif

	//showImage( imgSrc , 1 , "original" , 1000 );
	float confidence;
	IplImage * imgOut = cvCreateImage( cvSize(280 , 280) , 8 , 1 );
	cvSetZero( imgOut );
	
	int score = looksLikeNumber( imgSrc,   imgOut,   confidence , 0.05);

	std::cout << "  score = " << score << std::endl;
	std::cout << "  confidence = " << confidence << std::endl;


	cvReleaseImage( &imgSrc );
//	cvReleaseImage(&imgOut);

	deletePredictor();

	vector<pair<string, int> > imgName;

	int count = getAllImages(  imgName, "C:\\Users\\JohnHush\\Desktop\\finetune_data"  );

	printf( "number of images = %d\n" , count  );

	char s;

	std::cin >> s;

	return 1;
}

/*
int getAllImages(vector< pair<string, int> > & imgName, string path, int LABEL)
{
	struct _finddata_t fileinfo;
	long handle;
	string target;
	string s_tmp;
	
	if (path.size() > 1 && path.back() != '\\')
		target = path + '\\';

	target += "*.*";

	handle = _findfirst( target.c_str() , &fileinfo);
	int count = 0;
	
	if (handle == -1) return count;
	int label;
	do
	{
		if (LABEL == -1 && strcmp(fileinfo.name , ".") != 0 && 
			strcmp(fileinfo.name , "..") != 0  && (fileinfo.attrib &  _A_SUBDIR) )
		{
			if ( strlen(fileinfo.name) == 1 && fileinfo.name[0] >= '0' && fileinfo.name[0] <= '9' )
				label = fileinfo.name[0] - '0';
			else
				label = -1;

			count += getAllImages(imgName, s_tmp.assign(path).append("\\").append(fileinfo.name) , label);
		}
		if (LABEL != -1 && strcmp(fileinfo.name, ".") != 0 &&
			strcmp(fileinfo.name, "..") != 0 && !(fileinfo.attrib &  _A_SUBDIR))
		{
			int tmp = LABEL;
			if (strlen(fileinfo.name) >= 5 &&
				fileinfo.name[strlen(fileinfo.name) - 1] == 'g' &&
				fileinfo.name[strlen(fileinfo.name) - 2] == 'p' &&
				fileinfo.name[strlen(fileinfo.name) - 3] == 'j' &&
				fileinfo.name[strlen(fileinfo.name) - 4] == '.' &&
				fileinfo.name[strlen(fileinfo.name) - 5] != 'p' &&
				fileinfo.name[strlen(fileinfo.name) - 5] >= '0' &&
				fileinfo.name[strlen(fileinfo.name) - 5] <= '9'
				)
					tmp = fileinfo.name[strlen(fileinfo.name) - 5] - '0';

			count++;
			imgName.push_back(std::make_pair(s_tmp.assign(path).append("\\").append(fileinfo.name), tmp ));
		}
	} while (_findnext(handle, &fileinfo) == 0);

	return count;
}
*/

