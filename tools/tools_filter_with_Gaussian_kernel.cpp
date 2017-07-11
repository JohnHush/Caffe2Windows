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

int main()
{
	string target_path("F:/Lenet_training_set/collected_data-PROCESSED5/");
	string target_path2("F:/Lenet_training_set/filtered_with_gaussian_kernel/");


	for (int i = 0; i < 10; i++)
	{
		mkdir((target_path2 + std::to_string(i)).c_str());
	}

	for (int i = 0; i < 10; i++)
	{
		string target = target_path + std::to_string(i) + "/" + "*.*";

		struct _finddata_t fileinfo;
		long handle = _findfirst(target.c_str(), &fileinfo);
		if (handle == -1) continue;

		do
		{
			if (strcmp(fileinfo.name, ".") == 0 || strcmp(fileinfo.name, "..") == 0)
				continue;

			string picName = target_path + std::to_string(i) + "/" + fileinfo.name;

			IplImage * imgSrc = cvLoadImage( picName.c_str() , CV_LOAD_IMAGE_GRAYSCALE );

			IplImage * imgRst = cvCreateImage( cvSize( imgSrc->width , imgSrc->height) , 8 , 1 );

			cvSmooth( imgSrc , imgRst , 2 , 17 );

			uchar imgMax = 0;
			for (int ir = 0; ir < 280; ir++)
				for (int ic = 0; ic < 280; ic++)
					if (cvGetReal2D(imgRst, ir, ic) > imgMax)
						imgMax = cvGetReal2D(imgRst, ir, ic);

			float ratio = 255. / (imgMax+1);

			for (int ir = 0; ir < 280; ir++)
				for (int ic = 0; ic < 280; ic++)
					cvSetReal2D( imgRst , ir , ic , uchar(ratio * cvGetReal2D( imgRst , ir , ic )) );

			cvSaveImage((target_path2 + std::to_string(i) + "/" + fileinfo.name).c_str(), imgRst);


			cvReleaseImage(&imgSrc);
			cvReleaseImage(&imgRst);

		} while (_findnext(handle, &fileinfo) == 0);
	}
}