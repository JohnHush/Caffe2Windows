#include <caffe/proto/caffe.pb.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <fcntl.h>
#include "util_caffe.hpp"
#include "Binarizator/adaptive_threshold.hpp"
#include "tools_classifier.hpp"

using namespace std;
//using namespace caffe;

//#define FINETUNE

int main( int argc , char ** argv )
{
	string file_name;
#ifdef UNIX
	IplImage * imgtst = cvLoadImage( argv[1] , CV_LOAD_IMAGE_COLOR );
#endif
#ifdef _WINDOWS
	IplImage * imgtst = cvLoadImage( "D:\\MyProjects\\orion-eye\\test_data\\wrong_data\\444_2.png" , CV_LOAD_IMAGE_COLOR );
#endif


	//showImage(imgtst, 1, "red one", 0);
	if ( imgtst == NULL )
    {
        printf( "we don't get an image!\n " );
		return -1;
    }
    
	AdaThre adapt_thresholder( 201 , 20 );

	IplImage * imgred = cvCreateImage( cvSize(28,28) , 8 , 1 );
	cvSetZero( imgred );

	bool hasma = jh::getRedPixelsInHSVRange( imgtst , adapt_thresholder , 0.05 , imgred );

	//showImage( imgred , 10 , "red one" , 0 );
#ifdef DEBUG
	showImage( imgred , 10 , "red" );
#endif

	if ( !hasma )
	{
		cout << "the image is blank!\n";
		return -1;
	}

	ldb_handler MyDBHandler( "D:\\MyProjects\\orion-eye\\base_data\\finetune_training_data_leveldb" );
	MyDBHandler.showLastData();
	vector<IplImage *> imgs(1);
	vector<int> labels(1);
	imgs[0] = imgred;
	labels[0] = 6;
	MyDBHandler.addSomeData( imgs , labels );
	MyDBHandler.closeDB();
	MyDBHandler.showLastData();
	MyDBHandler.closeDB();
	MyDBHandler.resetDB();
	MyDBHandler.showLastData();
	return 0;

#ifdef UNIX
	string deployModel( "/home/pitaloveu/orion-eye/build/src_build/deploy_lenet.prototxt" );
	string caffeModel( "/home/pitaloveu/orion-eye/build/src_build/lenet_FINETUNE.caffemodel" );
#endif
#ifdef _WINDOWS
	string deployModel( "D:\\MyProjects\\orion-eye\\deploy_lenet.prototxt" );
	string caffeModel( "D:\\MyProjects\\orion-eye\\lenet_FINETUNE.caffemodel" );
#endif
//	vector<float> score = compute_score_by_caffe( imgred , deployModel , caffeModel );
	vector<float> score( 10, 0. );

	fstream input( caffeModel.c_str() , ios::in | ios::binary );
	
	caffe::NetParameter net;
	net.ParseFromIstream( &input );
	compute_score( imgred , net , score ); 

	for ( int i = 0 ; i < 10 ; i++)
		cout << "i = " << i << "  score = " << score[i] << endl;

	char sb;
	std::cin >> sb;
	return 0;
}
