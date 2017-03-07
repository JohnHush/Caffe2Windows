cmake_minimum_required( VERSION 3.0 )

project( TEST_IDENTIFY C CXX )

if ( APPLE )

include( /Users/JohnHush/caffe/build/install/share/Caffe/CaffeConfig.cmake )
add_definitions( -DAPPLE )
add_subdirectory( src src_build )

endif( APPLE )

if( MSVC )
include( D://caffe//scripts//build//libraries//caffe-builder-config.cmake )
include ( D://caffe//scripts//build//install//share//Caffe//CaffeConfig.cmake )

add_definitions( -DMSVC ${Caffe_DEFINITIONS} )
include_directories( ${Caffe_INCLUDE_DIRS} )
link_libraries( ${Caffe_LIBRARIES} )
link_libraries( ${OpenCV_LIBS} )
add_subdirectory( src src_build )

set( CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install )

endif( MSVC )
