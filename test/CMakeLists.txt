
include_directories( AFTER "../src" )

file( GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp )

foreach( source ${srcs} )
	get_filename_component( name ${source} NAME_WE )
	if( name MATCHES "test_280_280_pixels_Input" OR name MATCHES "test_OCRAPI_looksLikeNumber" )
		add_executable( ${name} ${source} ../src/caffe.pb.cc )
		target_link_libraries( ${name} OCR_PREDICT ${OCR_PREDICTION_LIB} )
		target_include_directories( ${name} PUBLIC ${OCR_PREDICTION_INCLUDE} )
		target_compile_definitions( ${name} PUBLIC ${OCR_PREDICTION_DEFINITION} )
	else()
		add_executable( ${name} ${source} )
		target_link_libraries( ${name} OCR_TRAIN ${OCR_TRAINING_LIB} )
		target_include_directories( ${name} PUBLIC ${OCR_TRAINING_INCLUDE} )
		target_compile_definitions( ${name} PUBLIC ${OCR_TRAINING_DEFINITION} )
endif()
endforeach(source)
