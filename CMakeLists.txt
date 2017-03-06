cmake_minimum_required( VERSION 3.0 )

project( TEST_IDENTIFY C CXX )

include( /Users/JohnHush/caffe/build/install/share/Caffe/CaffeConfig.cmake )

if( APPLE )
    add_definitions( -DAPPLE )
endif( )

add_subdirectory( src src_build )
