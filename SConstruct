env = Environment()
env.Append(CXXFLAGS = '-std=c++11')

Program( 'readCaffeModel' , [ 'readCaffeModel.cpp' , 'caffe.pb.cc' , 'Blob.cpp' , \
								'RedPixelsExtractor.cpp' , 'util.cpp'] , 
			LIBS = ['protobuf' , 'm' , 'cv' , 'highgui'] , 
			LIBPATH = [ '/usr/local/lib' , '/usr/lib' , '/usr/lib64' ])


