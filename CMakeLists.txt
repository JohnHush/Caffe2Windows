cmake_minimum_required( VERSION 3.0 )

project( TEST_IDENTIFY C CXX )

option( DEFINE_DEBUG "if adding #define DEBUG in the program " OFF )
if( DEFINE_DEBUG )
	add_definitions( -DDEBUG )
endif( DEFINE_DEBUG )

if ( UNIX )
	set( CMAKE_CXX_STANDARD 11)
	include( /home/pitaloveu/caffe/build/install/share/Caffe/CaffeConfig.cmake)
	include( /usr/local/share/OpenCV/OpenCVConfig.cmake )

	set( OCR_PREDICTION_LIB ${OpenCV_LIBS};/usr/lib64/libopenblas.so;/usr/local/lib/libprotoc.so )
	set( OCR_PREDICTION_INCLUDE ${OpenCV_INCLUDE_DIRS};/usr/include/openblas )
	set( OCR_PREDICTION_DEFINITION UNIX )

	set( OCR_TRAINING_LIB ${Caffe_LIBRARIES};${OpenCV_LIBS};/usr/lib64/libopenblas.so;/usr/local/lib/libprotoc.so )
	set( OCR_TRAINING_INCLUDE /usr/include/openblas;${Caffe_INCLUDE_DIRS};${OpenCV_INCLUDE_DIRS})
	set( OCR_TRAINING_DEFINITION UNIX CPU_ONLY USE_OPENCV USE_LMDB USE_LEVELDB)

	#add_definitions( -DUNIX ${Caffe_DEFINITIONS} )
	#include_directories( ${Caffe_INCLUDE_DIRS} )
	#include_directories( ${OpenCV_INCLUDE_DIRS} )
	#include_directories( BEFORE /usr/include/openblas )
	#link_libraries( ${Caffe_LIBRARIES} )
	#link_libraries( ${OpenCV_LIBS} )
	#link_libraries( /usr/lib64/libopenblas.so )
	#link_libraries( /usr/local/lib/libprotoc.so )
	#add_definitions( -DUNIX )
	add_subdirectory( src src_build )
	add_subdirectory( test test_build )
	add_subdirectory( tools tools_build )
endif()


if ( APPLE )

include( /Users/JohnHush/caffe/build/install/share/Caffe/CaffeConfig.cmake )
add_definitions( -DAPPLE )
add_subdirectory( src src_build )

endif( APPLE )

if( MSVC )
	# the OCR_PREDICTION_LIB should be built by MSVC2010, in win32 system;
	# the rely libs will contain :
	# 1. OPENCV_2.4.6
	# 2. OPENBLAS
	# 3. protobuf 3.2.0

	include ( D://MyLibs//VS2010LIB//opencv246//OpenCVConfig.cmake )
	set( OPENCV246_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} )
	set( OPENCV246_LIBS ${OpenCV_LIBS} )
	set( OPENBLAS_INCLUDE_DIRS D://MyLibs//VS2010LIB//OpenBLAS-v0.2.14-Win32//include )
	set( OPENBLAS_LIBS D://MyLibs//VS2010LIB//OpenBLAS-v0.2.14-Win32//lib//libopenblas.a )
	set( PROTOBUF_INCLUDE_DIRS D://MyLibs//VS2010LIB//protobuf//include )
	set( PROTOBUF_LIBS optimized D://MyLibs//VS2010LIB//protobuf//lib//libprotobuf.lib 
						optimized D://MyLibs//VS2010LIB//protobuf//lib//libprotobuf-lite.lib
						optimized D://MyLibs//VS2010LIB//protobuf//lib//libprotoc.lib 
						debug D://MyLibs//VS2010LIB//protobuf//lib//libprotobufd.lib 
						debug D://MyLibs//VS2010LIB//protobuf//lib//libprotobuf-lited.lib
						debug D://MyLibs//VS2010LIB//protobuf//lib//libprotocd.lib )

	set( OCR_PREDICTION_LIB ${OPENCV246_LIBS} ${OPENBLAS_LIBS} ${PROTOBUF_LIBS} )
	set( OCR_PREDICTION_INCLUDE ${OPENCV246_INCLUDE_DIRS} ${OPENBLAS_INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIRS} )
	set( OCR_PREDICTION_DEFINITION _WINDOWS )


	include( D://MyLibs//caffe-rely//caffe-builder-config.cmake )
	set(OPENCV_DIR "D:\\MyLibs\\opencv2_4_13"  CACHE PATH "")
	set(opencv_DIR "D:\\MyLibs\\opencv2_4_13"  CACHE PATH "")
	set(OpenCV_DIR "D:\\MyLibs\\opencv2_4_13"  CACHE PATH "")
	set(OpenCV_STATIC OFF CACHE BOOL "")

	include( D://MyLibs//caffe//share//Caffe//CaffeConfig.cmake )

	set( OCR_TRAINING_LIB ${Caffe_LIBRARIES};${OpenCV_LIBS} )
	set( OCR_TRAINING_INCLUDE ${Caffe_INCLUDE_DIRS};${OpenCV_INCLUDE_DIRS})
	set( OCR_TRAINING_DEFINITION _WINDOWS CPU_ONLY USE_OPENCV USE_LMDB USE_LEVELDB)

#add_definitions( ${Caffe_DEFINITIONS} )
#include_directories( ${Caffe_INCLUDE_DIRS} )
#link_libraries( ${Caffe_LIBRARIES} )
#link_libraries( ${OpenCV_LIBS} )
	add_subdirectory( src src_build )

set( CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install )


endif()

