set( SRC_LIST_LIB "" )

list( APPEND SRC_LIST_LIB HandWritingDigitsRecognitionSystem.cpp )
list( APPEND SRC_LIST_LIB  tools_classifier.cpp util.cpp )
list( APPEND SRC_LIST_LIB Binarizator/adaptive_threshold.cpp )
list( APPEND SRC_LIST_LIB Boxdetector/line_box_detector.cpp )
list( APPEND SRC_LIST_LIB Classifier/Mixed_Gaussian_Classifier.cpp caffe/proto/caffe.pb.cc )

add_library( OCR_PREDICT SHARED ${SRC_LIST_LIB} )

target_link_libraries( OCR_PREDICT PUBLIC ${OCR_PREDICTION_LIB} )
target_include_directories( OCR_PREDICT PUBLIC ${OCR_PREDICTION_INCLUDE} )
target_compile_definitions( OCR_PREDICT PUBLIC ${OCR_PREDICTION_DEFINITION} )

set( SRC_LIST_OCRTRAIN_LIB "" )

list( APPEND SRC_LIST_OCRTRAIN_LIB HandWritingDigitsRecognitionSystem.cpp util_caffe.cpp )
list( APPEND SRC_LIST_OCRTRAIN_LIB  tools_classifier.cpp util.cpp )
list( APPEND SRC_LIST_OCRTRAIN_LIB Binarizator/adaptive_threshold.cpp )
list( APPEND SRC_LIST_OCRTRAIN_LIB Boxdetector/line_box_detector.cpp )
list( APPEND SRC_LIST_OCRTRAIN_LIB Classifier/Mixed_Gaussian_Classifier.cpp caffe/proto/caffe.pb.cc )

add_library( OCR_TRAIN SHARED ${SRC_LIST_OCRTRAIN_LIB})

target_link_libraries( OCR_TRAIN PUBLIC ${OCR_TRAINING_LIB} )
target_include_directories( OCR_TRAIN PUBLIC ${OCR_TRAINING_INCLUDE} )
target_compile_definitions( OCR_TRAIN PUBLIC ${OCR_TRAINING_DEFINITION} )

INSTALL( TARGETS OCR_PREDICT OCR_TRAIN
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION staticlib
        ARCHIVE DESTINATION lib )
