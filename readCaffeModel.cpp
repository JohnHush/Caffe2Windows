#include "caffe.pb.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "Boxdetector/line_box_detector.hpp"
#include "caffe/caffe.hpp"

using namespace std;
using namespace caffe;

#include <fcntl.h>
#include "util.hpp"
#include "Binarizator/adaptive_threshold.hpp"
#include "tools_classifier.hpp"

int main( int argc , char ** argv )
{
	string name ( argv[1] );
	string file_name;
	file_name = "/home/pitaloveu/Desktop/" + name + ".png";
	IplImage * imgtst = cvLoadImage( file_name.c_str() , CV_LOAD_IMAGE_COLOR );

	const char * filename = "lenet_iter_200.caffemodel";
	caffe::NetParameter net;
	fstream input( filename , ios::in | ios::binary);	
	net.ParseFromIstream( &input );

	AdaThre adapt_thresholder( 201 , 20 );

	IplImage * imgred = cvCreateImage( cvSize(28,28) , 8 , 1 );
	cvSetZero( imgred );

	bool hasma = jh::getRedPixelsInHSVRange( imgtst , adapt_thresholder , 0.1 , imgred );

	if ( !hasma )
	{
		cout << "the image is blank!\n";
		return -1;
	}

	vector<float> score;
	compute_score( imgred , net , score );

	for ( int i = 0 ; i < 10 ; i++)
		cout << "i = " << i << "  score = " << score[i] << endl;

	cout << "end of calculating the score!\n" << endl;
	char s;
	cin >>s;

if ( 0 ) {
	SolverParameter solver_param;
	ReadSolverParamsFromTextFileOrDie( "lenet_solver_adam.prototxt" , &solver_param);
	solver_param.mutable_train_state()->set_level(0);
	Caffe::set_mode( Caffe::CPU );
	shared_ptr<caffe::Solver<float> >
		      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
	solver->net()->CopyTrainedLayersFrom( "_iter_5.caffemodel" );

	Blob<float>* input_layer0 = solver->net()->input_blobs()[0];
	Blob<float>* input_layer1 = solver->net()->input_blobs()[1];

	float * input_data0 = input_layer0->mutable_cpu_data();
	float * input_data1 = input_layer1->mutable_cpu_data();

	for ( int irow = 0 ; irow < 28 ; ++ irow )
    for ( int icol = 0 ; icol < 28 ; ++ icol )
		input_data0[28*irow + icol] = cvGetReal2D( imgred , irow , icol ) * 0.00390625;

//	for ( int i =0 ; i < 10 ; i++ )
//		input_data1[i] = 0;
	input_data1[0] = 0.;

	solver->Solve();

	return 0;
}
if (0 ){
	Caffe::set_mode( Caffe::CPU );
	shared_ptr<Net<float> > net_;
	net_.reset( new Net<float>( string("deploy_lenet.prototxt") , TEST ) );
	net_->CopyTrainedLayersFrom( "lenet_iter_200.caffemodel" );
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape( 1 , 1 , 28 , 28 );
	net_->Reshape();
	float * input_data = input_layer->mutable_cpu_data();
	for ( int irow = 0 ; irow < 28 ; ++ irow )
	for ( int icol = 0 ; icol < 28 ; ++ icol )
		input_data[28*irow + icol] = cvGetReal2D( imgred , irow , icol ) * 0.00390625;
	net_->Forward();
	Blob<float> * output_layer = net_->output_blobs()[0];
	const float * output_data = output_layer->cpu_data();
	for ( int i = 0 ; i < 10 ; i ++ )
		cout << "label= " << i << "  score = " << output_data[i] << endl;
}
	return 0;
}
