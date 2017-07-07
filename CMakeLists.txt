cmake_minimum_required( VERSION 3.0 )

project( OCR_TRAIN_AND_PREDICT C CXX )

option( DEFINE_DEBUG "if adding #define DEBUG in the program " OFF )
if( DEFINE_DEBUG )
	add_definitions( -DDEBUG )
endif( DEFINE_DEBUG )


# the OCR_PREDICTION_LIB should be built by MSVC2010, in win32 system;
# the rely libs will contain :
# 1. OPENCV_2.4.6
# 2. OPENBLAS
# 3. protobuf 3.2.0

set( PATH_TO_VS2010_OPENCV246 D:/MyLibs/VS2010LIB/opencv246 )
set( PATH_TO_VS2010_PROTOBUF320  D:/MyLibs/VS2010LIB/protobuf )
set( PATH_TO_VS2010_OPENBLAS0214 D:/MyLibs/VS2010LIB/OpenBLAS-v0.2.14-Win32 )
set( PATH_TO_VS2015_OPENCV2413 D:/MyLibs/opencv2_4_13 )
set( PATH_TO_VS2015_CAFFE_BUILDER D:/MyLibs/caffe-rely )
set( PATH_TO_VS2015_CAFFE D:/MyLibs/caffe )


set( OPENCV246_INCLUDE_DIRS ${PATH_TO_VS2010_OPENCV246}/include ${PATH_TO_VS2010_OPENCV246}/include/opencv )
set( OPENCV246_LIBS debug ${PATH_TO_VS2010_OPENCV246}/lib/opencv_core246d.lib
	debug ${PATH_TO_VS2010_OPENCV246}/lib/opencv_imgproc246d.lib
	debug ${PATH_TO_VS2010_OPENCV246}/lib/opencv_highgui246d.lib
	optimized ${PATH_TO_VS2010_OPENCV246}/lib/opencv_core246.lib
	optimized ${PATH_TO_VS2010_OPENCV246}/lib/opencv_highgui246.lib
	optimized ${PATH_TO_VS2010_OPENCV246}/lib/opencv_imgproc246.lib )

set( OPENBLAS_INCLUDE_DIRS ${PATH_TO_VS2010_OPENBLAS0214}/include )
set( OPENBLAS_LIBS ${PATH_TO_VS2010_OPENBLAS0214}/libopenblas.dll.a )

set( PROTOBUF_INCLUDE_DIRS ${PATH_TO_VS2010_PROTOBUF320}/include )
set( PROTOBUF_LIBS optimized ${PATH_TO_VS2010_PROTOBUF320}/lib//libprotobuf.lib 
	optimized ${PATH_TO_VS2010_PROTOBUF320}/lib/libprotobuf-lite.lib
	optimized ${PATH_TO_VS2010_PROTOBUF320}/lib/libprotoc.lib 
	debug ${PATH_TO_VS2010_PROTOBUF320}/lib/libprotobufd.lib 
	debug ${PATH_TO_VS2010_PROTOBUF320}/lib/libprotobuf-lited.lib
	debug ${PATH_TO_VS2010_PROTOBUF320}/lib/libprotocd.lib )

set( OCR_PREDICTION_LIB ${OPENCV246_LIBS} ${OPENBLAS_LIBS} ${PROTOBUF_LIBS} )
set( OCR_PREDICTION_INCLUDE ${OPENCV246_INCLUDE_DIRS} ${OPENBLAS_INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIRS} )
set( OCR_PREDICTION_DEFINITION _WINDOWS BUILD_OCR_PREDICT )


include( ${PATH_TO_VS2015_CAFFE_BUILDER}/caffe-builder-config.cmake )
set(OPENCV_DIR ${PATH_TO_VS2015_OPENCV2413}  CACHE PATH "")
set(opencv_DIR ${PATH_TO_VS2015_OPENCV2413}  CACHE PATH "")
set(OpenCV_DIR ${PATH_TO_VS2015_OPENCV2413}  CACHE PATH "")
set(OpenCV_STATIC OFF CACHE BOOL "")

include( ${PATH_TO_VS2015_CAFFE}/share/Caffe/CaffeConfig.cmake )

set( OCR_TRAINING_LIB ${Caffe_LIBRARIES};${OpenCV_LIBS} )
set( OCR_TRAINING_INCLUDE ${Caffe_INCLUDE_DIRS};${OpenCV_INCLUDE_DIRS})
set( OCR_TRAINING_DEFINITION _WINDOWS CPU_ONLY USE_OPENCV USE_LMDB USE_LEVELDB)


add_subdirectory( src src_build )
add_subdirectory( test test_build )
add_subdirectory( tools tools_build )

if( FALSE )

set( OPENCV246_INCLUDE_DIRS D:/MyLibs/VS2010LIB/opencv246/include D:/MyLibs/VS2010LIB/opencv246/include/opencv )
set( OPENCV246_LIBS debug D:/MyLibs/VS2010LIB/opencv246/lib/opencv_core246d.lib
	debug D:/MyLibs/VS2010LIB/opencv246/lib/opencv_imgproc246d.lib
	debug D:/MyLibs/VS2010LIB/opencv246/lib/opencv_highgui246d.lib
	optimized D:/MyLibs/VS2010LIB/opencv246/lib/opencv_core246.lib
	optimized D:/MyLibs/VS2010LIB/opencv246/lib/opencv_highgui246.lib
	optimized D:/MyLibs/VS2010LIB/opencv246/lib/opencv_imgproc246.lib )

set( OPENBLAS_INCLUDE_DIRS D://MyLibs//VS2010LIB//OpenBLAS-v0.2.14-Win32//include )
set( OPENBLAS_LIBS D://MyLibs//VS2010LIB//OpenBLAS-v0.2.14-Win32//lib//libopenblas.dll.a )

set( PROTOBUF_INCLUDE_DIRS D://MyLibs//VS2010LIB//protobuf//include )
set( PROTOBUF_LIBS optimized D://MyLibs//VS2010LIB//protobuf//lib//libprotobuf.lib 
	optimized D://MyLibs//VS2010LIB//protobuf//lib//libprotobuf-lite.lib
	optimized D://MyLibs//VS2010LIB//protobuf//lib//libprotoc.lib 
	debug D://MyLibs//VS2010LIB//protobuf//lib//libprotobufd.lib 
	debug D://MyLibs//VS2010LIB//protobuf//lib//libprotobuf-lited.lib
	debug D://MyLibs//VS2010LIB//protobuf//lib//libprotocd.lib )

set( OCR_PREDICTION_LIB ${OPENCV246_LIBS} ${OPENBLAS_LIBS} ${PROTOBUF_LIBS} )
set( OCR_PREDICTION_INCLUDE ${OPENCV246_INCLUDE_DIRS} ${OPENBLAS_INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIRS} )
set( OCR_PREDICTION_DEFINITION _WINDOWS BUILD_OCR_PREDICT )


include( D://MyLibs//caffe-rely//caffe-builder-config.cmake )
set(OPENCV_DIR "D:\\MyLibs\\opencv2_4_13"  CACHE PATH "")
set(opencv_DIR "D:\\MyLibs\\opencv2_4_13"  CACHE PATH "")
set(OpenCV_DIR "D:\\MyLibs\\opencv2_4_13"  CACHE PATH "")
set(OpenCV_STATIC OFF CACHE BOOL "")

include( D://MyLibs//caffe//share//Caffe//CaffeConfig.cmake )

set( OCR_TRAINING_LIB ${Caffe_LIBRARIES};${OpenCV_LIBS} )
set( OCR_TRAINING_INCLUDE ${Caffe_INCLUDE_DIRS};${OpenCV_INCLUDE_DIRS})
set( OCR_TRAINING_DEFINITION _WINDOWS CPU_ONLY USE_OPENCV USE_LMDB USE_LEVELDB)


add_subdirectory( src src_build )
add_subdirectory( test test_build )
add_subdirectory( tools tools_build )

endif()
