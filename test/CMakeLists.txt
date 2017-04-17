
include_directories( AFTER "../src" )

add_executable( test_predict test_dll.cpp )
target_link_libraries( test_predict OCR_PREDICT ${OCR_PREDICTION_LIB} )
target_include_directories( test_predict PUBLIC ${OCR_PREDICTION_INCLUDE} )
target_compile_definitions( test_predict PUBLIC ${OCR_PREDICTION_DEFINITION} )


add_executable( test_caffe_train test_caffe_train.cpp )

target_link_libraries( test_caffe_train PUBLIC ${OCR_TRAINING_LIB} OCR_TRAIN)
target_include_directories( test_caffe_train PUBLIC ${OCR_TRAINING_INCLUDE} )
target_compile_definitions( test_caffe_train PUBLIC ${OCR_TRAINING_DEFINITION} )

#add_executable( test_leveldb test_leveldb_read_write.cpp )

#target_link_libraries( test_leveldb PUBLIC ${OCR_TRAINING_LIB} OCR_TRAIN)
#target_include_directories( test_leveldb PUBLIC ${OCR_TRAINING_INCLUDE} )
#target_compile_definitions( test_leveldb PUBLIC ${OCR_TRAINING_DEFINITION} )
