set( SRC_LIST "" )

list( APPEND SRC_LIST MAINENTRY.cpp tools_classifier.cpp util.cpp )
list( APPEND SRC_LIST Binarizator/adaptive_threshold.cpp )
list( APPEND SRC_LIST Boxdetector/line_box_detector.cpp )
list( APPEND SRC_LIST Classifier/Mixed_Gaussian_Classifier.cpp )

#set( CMAKE_BUILD_TYPE Release )

add_executable( MAINENTRY ${SRC_LIST} )

#target_link_libraries( MAINENTRY ${OpenBLAS_LIB} ${Caffe_LIBRARIES} ${OpenCV_LIBS} )

if ( MSVC )
set( SRC_LIST_LIB "" )

list( APPEND SRC_LIST_LIB Boxdetector/line_box_detector.hpp )
list( APPEND SRC_LIST_LIB HandWritingDigitsRecognitionSystem.cpp  tools_classifier.cpp util.cpp )
list( APPEND SRC_LIST_LIB Binarizator/adaptive_threshold.cpp )
list( APPEND SRC_LIST_LIB Boxdetector/line_box_detector.cpp )
list( APPEND SRC_LIST_LIB Classifier/Mixed_Gaussian_Classifier.cpp )

add_library( OCRLIB SHARED ${SRC_LIST_LIB} )

target_link_libraries( OCRLIB ${Caffe_LIBRARIES} ${OpenCV_LIBS} )


INSTALL( TARGETS OCRLIB
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION staticlib
        ARCHIVE DESTINATION lib )



add_executable( DLLTEST test_dll.cpp )
target_link_libraries( DLLTEST OCRLIB )

endif( MSVC )
