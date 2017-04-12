set( SRC_LIST "" )

include_directories( AFTER "../src" )
list( APPEND SRC_LIST convert_data_written_by_fengchejiaoyu.cpp )

add_executable( CONVERT_DATA ${SRC_LIST} )

target_link_libraries( CONVERT_DATA PUBLIC ${OCR_TRAINING_LIB} OCR_TRAIN )
target_include_directories( CONVERT_DATA PUBLIC ${OCR_TRAINING_INCLUDE} )
target_compile_definitions( CONVERT_DATA PUBLIC ${OCR_TRAINING_DEFINITION} )

INSTALL( TARGETS CONVERT_DATA
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION staticlib
        ARCHIVE DESTINATION lib )
