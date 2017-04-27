#include "util.hpp"
#include "util_caffe.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
#include <fstream>
#include "config.hpp"
#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include "util.hpp"
#include <caffe/caffe.hpp>
#ifdef _WINDOWS
#include <io.h>
#endif

#ifdef BUILD_OCR_PREDICT
#include "caffe.pb.h"
#else
#include <caffe/proto/caffe.pb.h>
#endif

using std::cout;
using std::endl;

#ifdef _WINDOWS
int getAllImages(vector< pair<string, int> > & imgName, string path, int LABEL)
{
	struct _finddata_t fileinfo;
	long handle;
	string target;
	string s_tmp;

	if (path.size() > 1 && path.back() != '\\')
		target = path + '\\';

	target += "*.*";

	handle = _findfirst(target.c_str(), &fileinfo);
	int count = 0;

	if (handle == -1) return count;
	int label;
	do
	{
		if (LABEL == -1 && strcmp(fileinfo.name, ".") != 0 &&
			strcmp(fileinfo.name, "..") != 0 && (fileinfo.attrib &  _A_SUBDIR))
		{
			if (strlen(fileinfo.name) == 1 && fileinfo.name[0] >= '0' && fileinfo.name[0] <= '9')
				label = fileinfo.name[0] - '0';
			else
				label = -1;

			count += getAllImages(imgName, s_tmp.assign(path).append("\\").append(fileinfo.name), label);
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
			imgName.push_back(std::make_pair(s_tmp.assign(path).append("\\").append(fileinfo.name), tmp));
		}
	} while (_findnext(handle, &fileinfo) == 0);

	return count;
}

OCRAPI void read_Windows_Data2_Existing_LevelDB(string data_path, ldb_handler & HANDLER )
{
	vector<pair<string, int> > imgName;

	int count = getAllImages(imgName, data_path);

//	ldb_handler MyDBHandler( lmdb_path );
	vector<cv::Mat> imgs;
	vector<int> labels;

	for (int item_id = 0; item_id < imgName.size(); ++item_id )
	{
		cv::Mat imgMat = cv::imread(imgName[item_id].first, CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat imgMatTmp( 28 , 28 , CV_8UC1 );

		if (imgMat.size() != cv::Size(280, 280))
		{
			LOG(INFO) << "some unqualified image occured" << endl;
			continue;
		}
		for (int irow = 0; irow < 28; irow++)
			for (int icol = 0; icol < 28; icol++)
				imgMatTmp.at<uchar>(irow, icol) = imgMat.at<uchar>(irow*10,icol*10);

		imgs.push_back( imgMatTmp );
		labels.push_back( imgName[item_id].second );
	}
	
//	imgs[0] = cv::Mat(imgred);
//	labels[0] = 6;
	HANDLER.addSomeData(imgs, labels);
//	MyDBHandler.closeDB();
//	MyDBHandler.showLastData();
	HANDLER.closeDB();
//	MyDBHandler.resetDB();
//	MyDBHandler.showLastData();
}

void read_Windows_Data2_LevelDB( string data_path , string lmdb_path )
{
	vector<pair<string, int> > imgName;

	int count = getAllImages(imgName, data_path );

	leveldb::DB* db;
	leveldb::Options options;
	leveldb::WriteOptions write_options;
	leveldb::Status status;

	options.create_if_missing = true;
	options.error_if_exists = true;
	write_options.sync = true;
	status = leveldb::DB::Open(options, lmdb_path , &db);
	if (!status.ok())
		LOG(FATAL) << "can't open the db " << endl;

	leveldb::WriteBatch* batch = NULL;
	batch = new leveldb::WriteBatch();

	char * pixels = new char[28 * 28];
	char label;

	string value;
	char key_cstr[10];
	count = 0;

	caffe::Datum datum;
	datum.set_channels(1);
	datum.set_height(28);
	datum.set_width(28);

	for (int item_id = 0; item_id < imgName.size(); ++item_id)
	{
		cv::Mat imgMat = cv::imread(imgName[item_id].first, CV_LOAD_IMAGE_GRAYSCALE);

		if (imgMat.size() != cv::Size(280, 280)) {
			LOG(INFO) << "some unqualified image occured" << endl;
			continue;
		}
		for (int irow = 0; irow < 28; irow++)
			for (int icol = 0; icol < 28; icol++)
				pixels[irow * 28 + icol] = imgMat.at<unsigned char>(irow * 10, icol * 10);

		datum.set_data(pixels, 28 * 28);
		datum.set_label(imgName[item_id].second);

		datum.SerializeToString(&value);
		snprintf(key_cstr, 10, "%08d", item_id);
		string keystr(key_cstr);
		batch->Put(keystr, value);

		if (++count % 1000 == 0)
		{
			db->Write(write_options, batch);
			delete batch;
			batch = new leveldb::WriteBatch();
		}
	}
	if (count % 1000 != 0)
	{
		db->Write(write_options, batch);
		delete batch;
		delete db;
	}
	delete[] pixels;
}
#endif

int ldb_handler::BASE_NUMBER_;


void ldb_handler::splitDB(string & training_set, string & test_set)
{
	DB_OPEN_OR_NOT pre_state = state_;
	if (state_ == CLOSE)
	{
		status_ = leveldb::DB::Open(options_, db_name_.c_str(), &db_);
		if (!status_.ok())	LOG(FATAL) << "cannot ok the db for some unknown reasons" << endl;
		it_ = db_->NewIterator(leveldb::ReadOptions());
		state_ = OPEN;
	}

	leveldb::DB * db1;
	leveldb::DB * db2;
//	leveldb::Options options_;
//	leveldb::WriteOptions write_options_;
//	leveldb::Status status_;
//	leveldb::Iterator * it_;
//	string db_name_;
	leveldb::Options options_SPLIT;
	leveldb::Status status_SPLIT;
//	leveldb::Iterator * it_SPLIT = db_->NewIterator(leveldb::ReadOptions());
//	it_ = db_->NewIterator(leveldb::ReadOptions());
	options_SPLIT.create_if_missing = true;
	options_SPLIT.error_if_exists = true;
	status_SPLIT = leveldb::DB::Open( options_SPLIT , training_set , &db1 );
	if (!status_SPLIT.ok())	LOG(FATAL) << "cannot ok the db for some unknown reasons" << endl;
	status_SPLIT = leveldb::DB::Open(options_SPLIT, test_set, &db2);
	if (!status_SPLIT.ok())	LOG(FATAL) << "cannot ok the db for some unknown reasons" << endl;

//	string value;
//	vector<string> db_KEYS;
	int count = 0;
	for (it_->SeekToFirst(); it_->Valid(); it_->Next())
	{		
		string value;
		status_SPLIT = db_->Get( leveldb::ReadOptions() , it_->key().ToString() , & value );
		if (status_SPLIT.ok())
		{
			if (rand() % 100 >= 20)
				db1->Put(leveldb::WriteOptions(), it_->key().ToString() , value );
				//db1->Put(leveldb::WriteOptions(), it_->key(), it_->value().ToString());
			else
				db2->Put(leveldb::WriteOptions(), it_->key().ToString(), value);
				//db2->Put(leveldb::WriteOptions(), it_->key(), it_->value().ToString());
		}
//		db_KEYS.push_back( it_->key().ToString() );

	}

	//cout << "db length = " << count << endl;
	delete db1;
	delete db2;
//	delete it_SPLIT;

	if (pre_state == CLOSE)
	{
		delete it_;
		delete db_;
		state_ = CLOSE;
	}
}
ldb_handler::ldb_handler( const string & db_name )
{
	BASE_NUMBER_ = 1840;
	db_name_ = db_name;
	options_.create_if_missing = false;
	write_options_.sync = true;
	status_ = leveldb::DB::Open( options_ , db_name.c_str() , &db_ );
	if ( !status_.ok() )	LOG(FATAL) << "cannot ok the db for some unknown reasons" << endl;
	it_ = db_->NewIterator( leveldb::ReadOptions() );
	state_ = OPEN;
}

void ldb_handler::addData( cv::Mat &imgAdd , int label , int key_index )
{
	caffe::Datum datum;
	datum.set_channels(1);
	datum.set_height(28);
	datum.set_width(28);
    char * pixels = new char [ 28 * 28 ];

    for ( int irow = 0 ; irow < 28 ; irow ++ )
    for ( int icol = 0 ; icol < 28 ; icol ++ )
       // pixels[ irow * 28 + icol ] = cvGetReal2D( imgAdd , irow , icol );
		pixels[irow * 28 + icol] = imgAdd.at<unsigned char>( irow , icol );

    datum.set_data( pixels , 28 *28 );
    datum.set_label( label );

    char key_cstr[10];
    string value;
    snprintf( key_cstr , 10 , "%08d" , key_index );
    datum.SerializeToString( &value );
    string keystr( key_cstr );

    db_->Put( write_options_ , keystr , value );

	delete [] pixels;
}
void ldb_handler::addSomeData( vector<cv::Mat> & imgAdds , vector<int> & labels )
{
	DB_OPEN_OR_NOT pre_state = state_;
	if ( state_ == CLOSE )
	{
		status_ = leveldb::DB::Open( options_ , db_name_.c_str() , &db_ );
		if ( !status_.ok() )	LOG(FATAL) << "cannot ok the db for some unknown reasons" << endl;
		it_ = db_->NewIterator( leveldb::ReadOptions() );
		state_ = OPEN;
	}

	CHECK_EQ( imgAdds.size() , labels.size() ) << "img number should equal to label number!" << endl;

	for ( int i = 0 ; i < labels.size() ; ++ i )
		addData( imgAdds[i] , labels[i] , BASE_NUMBER_ + i );

	if ( pre_state == CLOSE )
	{
		delete it_;
		delete db_;
		state_ = CLOSE;
	}
}

void ldb_handler::showLastData()
{
	DB_OPEN_OR_NOT pre_state = state_;
	if ( state_ == CLOSE )
	{
		status_ = leveldb::DB::Open( options_ , db_name_.c_str() , &db_ );
		if ( !status_.ok() )	LOG(FATAL) << "cannot ok the db for some unknown reasons" << endl;
		it_ = db_->NewIterator( leveldb::ReadOptions() );
		state_ = OPEN;
	}
	//IplImage * img = cvCreateImage( cvSize( 28 , 28 ) , 8 , 1 );
	cv::Mat img( 28 , 28 , CV_8UC1 );
	it_->SeekToLast();
	caffe::Datum datum;
	datum.ParseFromString( it_->value().ToString() );

	for ( int irow = 0 ; irow < 28 ; ++ irow ) 
	for ( int icol = 0 ; icol < 28 ; ++ icol ) 
	{
		int   data_index = irow * 28 + icol;
		float data_value = static_cast<float>(static_cast<uint8_t>( (datum.data()) [data_index]));
		//cvSetReal2D( img , irow , icol , data_value );
		img.at<uchar>(irow, icol) = uchar(data_value);
	}
	showImageMat( img , 10 , "LastImage" , 1000 );
//	cvReleaseImage( &img );

	if ( pre_state == CLOSE )
	{
		delete it_;
		delete db_;
		state_ = CLOSE;
	}
}

void ldb_handler::resetDB()
{
	DB_OPEN_OR_NOT pre_state = state_;

	if ( state_ == CLOSE )
	{
		status_ = leveldb::DB::Open( options_ , db_name_.c_str() , &db_ );
		if ( !status_.ok() )	LOG(FATAL) << "cannot ok the db for some unknown reasons" << endl;
		it_ = db_->NewIterator( leveldb::ReadOptions() );
		state_ = OPEN;
	}

	it_->SeekToLast();
	leveldb::WriteBatch batch;
	while( it_->key().ToString() != string("00001839") && it_->Valid() )
	{
		batch.Delete( it_->key() );
		it_->Prev();
	}
	status_ = db_->Write( write_options_ , &batch );
	if ( !status_.ok() )
		LOG(FATAL) << "someting wrong in writing the batch" <<endl;

	if ( pre_state == CLOSE )
	{
		delete it_;
		delete db_;
		state_ = CLOSE;
	}
}

void ldb_handler::closeDB()
{
	if ( state_ == OPEN )
	{
		state_ = CLOSE;
		delete it_;
		delete db_;
	}
}

vector<float> compute_score_by_caffe( const IplImage * imgSrc , const string & deploy_model , const string & caffe_model )
{
	using namespace caffe;

	Caffe::set_mode( Caffe::CPU );
	shared_ptr<Net<float> > net_;
	net_.reset( new Net<float>( deploy_model , TEST ) );
	net_->CopyTrainedLayersFrom( caffe_model );
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape( 1 , 1 , 28 , 28 );
	net_->Reshape();
	float * input_data = input_layer->mutable_cpu_data();
	for ( int irow = 0 ; irow < 28 ; ++ irow )
	for ( int icol = 0 ; icol < 28 ; ++ icol )
		input_data[28*irow + icol] = cvGetReal2D( imgSrc , irow , icol ) * 0.00390625;
	net_->Forward();
	Blob<float> * output_layer = net_->output_blobs()[0];
	const float * output_data = output_layer->cpu_data();
	vector<float> score( 10 );
	for ( int i = 0 ; i < 10 ; i ++ )
		score[i] = output_data[i];

	return score;
}

void finetune_with_Existing_LevelDB(const string & pretrained_model, const string & train_net_arch_prototxt)
{
	using namespace caffe;

	SolverParameter solver_param;

#ifdef UNIX
	int MAXBUFSIZE = 1024;
	int count;
	char buf[MAXBUFSIZE];

	count = readlink("/proc/self/exe", buf, MAXBUFSIZE);
	if (count < 0 || count >= MAXBUFSIZE)
		LOG(FATAL) << "size of the exe path wrong !" << std::endl;

	string exePath(buf);
	exePath = exePath.substr(0, exePath.rfind('/') + 1);
	LOG(INFO) << exePath << std::endl;
#endif

#ifdef _WINDOWS
	CHAR exeFullPath[MAX_PATH];
	string exePath;
	GetModuleFileNameA(NULL, exeFullPath, MAX_PATH);
	exePath = exeFullPath;
	exePath = exePath.substr(0, exePath.rfind('\\') + 1);
#endif

#ifdef APPLE
	LOG(FATAL) << "implement the method here" << std::endl;
#endif
	const int iteration_times = 100;
	stringstream ss;
	string s_iteration_times;
	ss << iteration_times;
	ss >> s_iteration_times;

	solver_param.set_net((exePath + train_net_arch_prototxt).c_str());
	solver_param.add_test_iter(10);
	solver_param.set_test_interval(10);
	solver_param.set_base_lr(0.0001);
	solver_param.set_momentum(0.9);
	solver_param.set_momentum2(0.999);
	solver_param.set_lr_policy("fixed");
	solver_param.set_display(0);
	solver_param.set_max_iter(iteration_times);
	solver_param.set_snapshot(iteration_times);
	solver_param.set_snapshot_prefix("retrained");
	solver_param.set_type("Adam");
	solver_param.set_solver_mode(SolverParameter::CPU);

	string name_retrained_model_by_caffe(solver_param.snapshot_prefix() +
		"_iter_" + s_iteration_times + ".caffemodel");

	string name_retrained_solvestate_by_caffe(solver_param.snapshot_prefix() +
		"_iter_" + s_iteration_times + ".solverstate");

	//	ReadSolverParamsFromTextFileOrDie( solver_prototxt , &solver_param);
	solver_param.mutable_train_state()->set_level(0);
	Caffe::set_mode(Caffe::CPU);
	shared_ptr<caffe::Solver<float> >
		solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

	solver->net()->CopyTrainedLayersFrom(exePath + pretrained_model);

	solver->Solve();

	if (std::remove((exePath + pretrained_model).c_str()) != 0)
		LOG(FATAL) << "cannot remove pretrained model for unknown reason!" << std::endl;
	if (std::rename(name_retrained_model_by_caffe.c_str(), (exePath + pretrained_model).c_str()) != 0)
		LOG(FATAL) << "unable to change the file name of the trained model for unknown reason!" << std::endl;
	if (std::remove(name_retrained_solvestate_by_caffe.c_str()) != 0)
		LOG(FATAL) << "unable to delete the solverstate file for unknown reason!" << std::endl;
}

void finetune_by_caffe_leveldb(const string & pretrained_model, const string & train_net_arch_prototxt, 
									vector< cv::Mat > & imgs, vector<int> & labels , const string & base_db )
{
	using namespace caffe;

	SolverParameter solver_param;

	ldb_handler MyDBHandler(base_db);
	MyDBHandler.resetDB();
	MyDBHandler.addSomeData(imgs, labels);
	MyDBHandler.closeDB();


#ifdef UNIX
	int MAXBUFSIZE = 1024;
	int count;
	char buf[MAXBUFSIZE];

	count = readlink("/proc/self/exe", buf, MAXBUFSIZE);
	if (count < 0 || count >= MAXBUFSIZE)
		LOG(FATAL) << "size of the exe path wrong !" << std::endl;

	string exePath(buf);
	exePath = exePath.substr(0, exePath.rfind('/') + 1);
	LOG(INFO) << exePath << std::endl;
#endif

#ifdef _WINDOWS
	CHAR exeFullPath[MAX_PATH];
	string exePath;
	GetModuleFileNameA(NULL, exeFullPath, MAX_PATH);
	exePath = exeFullPath;
	exePath = exePath.substr(0, exePath.rfind('\\') + 1);
#endif

#ifdef APPLE
	LOG(FATAL) << "implement the method here" << std::endl;
#endif
	const int iteration_times = 10;
	stringstream ss;
	string s_iteration_times;
	ss << iteration_times;
	ss >> s_iteration_times;

	solver_param.set_net((exePath + train_net_arch_prototxt).c_str());
	solver_param.add_test_iter(0);
	solver_param.set_test_interval(500);
	solver_param.set_base_lr(0.0001);
	solver_param.set_momentum(0.9);
	solver_param.set_momentum2(0.999);
	solver_param.set_lr_policy("fixed");
	solver_param.set_display(0);
	solver_param.set_max_iter(iteration_times);
	solver_param.set_snapshot(iteration_times);
	solver_param.set_snapshot_prefix("retrained");
	solver_param.set_type("Adam");
	solver_param.set_solver_mode(SolverParameter::CPU);

	string name_retrained_model_by_caffe(solver_param.snapshot_prefix() +
		"_iter_" + s_iteration_times + ".caffemodel");

	string name_retrained_solvestate_by_caffe(solver_param.snapshot_prefix() +
		"_iter_" + s_iteration_times + ".solverstate");
	
	//	ReadSolverParamsFromTextFileOrDie( solver_prototxt , &solver_param);
	solver_param.mutable_train_state()->set_level(0);
	Caffe::set_mode(Caffe::CPU);
	shared_ptr<caffe::Solver<float> >
		solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
	
	solver->net()->CopyTrainedLayersFrom(exePath + pretrained_model);
	
	solver->Solve();

	if (std::remove((exePath + pretrained_model).c_str()) != 0)
		LOG(FATAL) << "cannot remove pretrained model for unknown reason!" << std::endl;
	if (std::rename(name_retrained_model_by_caffe.c_str(), (exePath + pretrained_model).c_str()) != 0)
		LOG(FATAL) << "unable to change the file name of the trained model for unknown reason!" << std::endl;
	if (std::remove(name_retrained_solvestate_by_caffe.c_str()) != 0)
		LOG(FATAL) << "unable to delete the solverstate file for unknown reason!" << std::endl;

}

void finetune_by_caffe( const string & pretrained_model , const string & train_net_arch_prototxt , const IplImage * imgSrc , const int label )
// we gonna have two functions, 
// 1 : finetune()
//    this function just finetune the pretrained model previously
// 2 : get_back()
{
	using namespace caffe;

	SolverParameter solver_param;

#ifdef UNIX
	int MAXBUFSIZE = 1024;
	int count;
	char buf[MAXBUFSIZE];

	count = readlink( "/proc/self/exe" , buf , MAXBUFSIZE );
	if ( count < 0 || count >= MAXBUFSIZE )
		LOG(FATAL) << "size of the exe path wrong !" << std::endl;

	string exePath( buf );
	exePath = exePath.substr( 0 , exePath.rfind( '/' ) + 1 );
	LOG(INFO) << exePath << std::endl;
#endif

#ifdef _WINDOWS
	CHAR exeFullPath[MAX_PATH];
	string exePath;
	GetModuleFileNameA(NULL, exeFullPath, MAX_PATH);
	exePath = exeFullPath;
	exePath = exePath.substr(0, exePath.rfind('\\') + 1);
#endif

#ifdef APPLE
	LOG(FATAL) << "implement the method here" << std::endl;
#endif
	const int iteration_times = 10;
	stringstream ss;
	string s_iteration_times;
	ss << iteration_times;
	ss >> s_iteration_times;

	solver_param.set_net( (exePath + train_net_arch_prototxt ).c_str() );
	solver_param.add_test_iter( 0 );
	solver_param.set_test_interval( 500 );
	solver_param.set_base_lr( 0.0001 );
	solver_param.set_momentum( 0.9 );
	solver_param.set_momentum2( 0.999 );
	solver_param.set_lr_policy( "fixed" );
	solver_param.set_display( 0 );
	solver_param.set_max_iter( iteration_times );
	solver_param.set_snapshot( iteration_times );
	solver_param.set_snapshot_prefix( "retrained" );
	solver_param.set_type( "Adam" );
	solver_param.set_solver_mode( SolverParameter::CPU );

	string name_retrained_model_by_caffe( solver_param.snapshot_prefix() + 
			"_iter_" + s_iteration_times + ".caffemodel");

	string name_retrained_solvestate_by_caffe( solver_param.snapshot_prefix() + 
			"_iter_" + s_iteration_times + ".solverstate");

//	ReadSolverParamsFromTextFileOrDie( solver_prototxt , &solver_param);
	solver_param.mutable_train_state()->set_level(0);
	Caffe::set_mode( Caffe::CPU );
	shared_ptr<caffe::Solver<float> >
              solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
	solver->net()->CopyTrainedLayersFrom( exePath + pretrained_model );

	Blob<float>* input_layer0 = solver->net()->input_blobs()[0];
	Blob<float>* input_layer1 = solver->net()->input_blobs()[1];

	float * input_data0 = input_layer0->mutable_cpu_data();
	float * input_data1 = input_layer1->mutable_cpu_data();

	for ( int irow = 0 ; irow < 28 ; ++ irow )
	for ( int icol = 0 ; icol < 28 ; ++ icol )
		input_data0[28*irow + icol] = cvGetReal2D( imgSrc , irow , icol ) * 0.00390625;

	input_data1[0] = float(label);

	solver->Solve();

	if ( std::remove( ( exePath + pretrained_model ).c_str() ) != 0 )
		LOG(FATAL) << "cannot remove pretrained model for unknown reason!" << std::endl;
	if ( std::rename( name_retrained_model_by_caffe.c_str() , (exePath + pretrained_model).c_str() ) != 0 )
		LOG(FATAL) << "unable to change the file name of the trained model for unknown reason!" << std::endl;
	if ( std::remove( name_retrained_solvestate_by_caffe.c_str() )!= 0 ) 
		LOG(FATAL) << "unable to delete the solverstate file for unknown reason!" << std::endl;
}

void getback_to_ORIGINAL_MODEL( const string & pretrained_model , const string & ori_model )
{
#ifdef _WINDOWS
	CHAR exeFullPath[MAX_PATH];
	string exePath;
	GetModuleFileNameA( NULL , exeFullPath , MAX_PATH );
	exePath = exeFullPath;
	exePath = exePath.substr( 0 , exePath.rfind('\\') + 1 );
#endif
#ifdef APPLE
	LOG(FATAL) << "need to be implemented in mac os" << std::endl;
#endif
#ifdef UNIX
	int MAXBUFSIZE = 1024;
    int count;
    char buf[MAXBUFSIZE];

    count = readlink( "/proc/self/exe" , buf , MAXBUFSIZE );
    if ( count < 0 || count >= MAXBUFSIZE )
        LOG(FATAL) << "size of the exe path wrong !" << std::endl;

    string exePath( buf );
    exePath = exePath.substr( 0 , exePath.rfind( '/' ) + 1 );
    LOG(INFO) << exePath << std::endl;
#endif

	if ( std::remove( ( exePath + pretrained_model ).c_str() ) != 0 )
		LOG(FATAL) << "cannot remove pretrained model for unknown reason" << std::endl;

	std::fstream ORI_FILE( (exePath + ori_model).c_str() , std::ios::in|std::ios::binary );
	std::fstream PRT_FILE( (exePath + pretrained_model).c_str() , std::ios::out|std::ios::binary );

	if( ! ORI_FILE || ! PRT_FILE )
	{
		ORI_FILE.close();
		PRT_FILE.close();
		LOG(FATAL) << "cannot open file stream " << std::endl;
	}
	char tmp;
	while( ORI_FILE.get(tmp) )
		PRT_FILE << tmp;

	ORI_FILE.close();
	PRT_FILE.close();
}
