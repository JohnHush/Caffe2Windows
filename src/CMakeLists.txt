set( SRC_LIST "" )

list( APPEND SRC_LIST MAINENTRY.cpp tools_classifier.cpp util.cpp )
list( APPEND SRC_LIST Binarizator/adaptive_threshold.cpp )
list( APPEND SRC_LIST Boxdetector/line_box_detector.cpp )
list( APPEND SRC_LIST Classifier/Mixed_Gaussian_Classifier.cpp )


add_executable( MAINENTRY ${SRC_LIST} )

target_link_libraries( MAINENTRY ${Caffe_LIBRARIES} )
