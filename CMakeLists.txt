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
include( D://MyLibs//caffe-rely//caffe-builder-config.cmake )
set(OPENCV_DIR "D:\\MyLibs\\opencv2_4_13"  CACHE PATH "")
set(opencv_DIR "D:\\MyLibs\\opencv2_4_13"  CACHE PATH "")
set(OpenCV_DIR "D:\\MyLibs\\opencv2_4_13"  CACHE PATH "")
set(OpenCV_STATIC OFF CACHE BOOL "")

include( D://MyLibs//caffe//share//Caffe//CaffeConfig.cmake )

add_definitions( ${Caffe_DEFINITIONS} )
include_directories( ${Caffe_INCLUDE_DIRS} )
link_libraries( ${Caffe_LIBRARIES} )
link_libraries( ${OpenCV_LIBS} )
add_subdirectory( src src_build )

set( CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install )

endif()

