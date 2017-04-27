
include_directories( AFTER "../src" )

add_executable( test_predict test_dll.cpp ../src/caffe.pb.cc )
target_link_libraries( test_predict OCR_PREDICT ${OCR_PREDICTION_LIB} )
target_include_directories( test_predict PUBLIC ${OCR_PREDICTION_INCLUDE} )
target_compile_definitions( test_predict PUBLIC ${OCR_PREDICTION_DEFINITION} )

add_executable( test_predict_using_caffe test_predict_using_caffe.cpp ../src/caffe.pb.cc )
target_link_libraries( test_predict_using_caffe OCR_PREDICT ${OCR_PREDICTION_LIB} )
target_include_directories( test_predict_using_caffe PUBLIC ${OCR_PREDICTION_INCLUDE} )
target_compile_definitions( test_predict_using_caffe PUBLIC ${OCR_PREDICTION_DEFINITION} )

add_executable( test_Windows_data2_LMDB test_addData_windows.cpp )
target_link_libraries( test_Windows_data2_LMDB PUBLIC ${OCR_TRAINING_LIB} OCR_TRAIN)
target_include_directories( test_Windows_data2_LMDB PUBLIC ${OCR_TRAINING_INCLUDE} )
target_compile_definitions( test_Windows_data2_LMDB PUBLIC ${OCR_TRAINING_DEFINITION} )


if( FALSE)
link_libraries(  ${OCR_TRAINING_LIB} OCR_TRAIN )
include_directories( ${OCR_TRAINING_INCLUDE} )
#add_definitions( ${OCR_TRAINING_DEFINITION} )

add_executable( test_caffe_train test_caffe_train.cpp )

#target_link_libraries( test_caffe_train PUBLIC ${OCR_TRAINING_LIB} OCR_TRAIN)
#target_include_directories( test_caffe_train PUBLIC ${OCR_TRAINING_INCLUDE} )
target_compile_definitions( test_caffe_train PUBLIC ${OCR_TRAINING_DEFINITION} )

add_executable( test_train_add_data_leveldb test_train_adding_data_2leveldb.cpp )

#target_link_libraries( test_train_add_data_leveldb PUBLIC ${OCR_TRAINING_LIB} OCR_TRAIN)
#target_include_directories( test_train_add_data_leveldb PUBLIC ${OCR_TRAINING_INCLUDE} )
target_compile_definitions( test_train_add_data_leveldb PUBLIC ${OCR_TRAINING_DEFINITION} )

add_executable( test_finetune_with_base_db  test_finetune_with_base_db.cpp )

#target_link_libraries( test_train_add_data_leveldb PUBLIC ${OCR_TRAINING_LIB} OCR_TRAIN)
#target_include_directories( test_train_add_data_leveldb PUBLIC ${OCR_TRAINING_INCLUDE} )
target_compile_definitions( test_train_add_data_leveldb PUBLIC ${OCR_TRAINING_DEFINITION} )


#add_executable( test_leveldb test_leveldb_read_write.cpp )

#target_link_libraries( test_leveldb PUBLIC ${OCR_TRAINING_LIB} OCR_TRAIN)
#target_include_directories( test_leveldb PUBLIC ${OCR_TRAINING_INCLUDE} )
#target_compile_definitions( test_leveldb PUBLIC ${OCR_TRAINING_DEFINITION} )

endif()
