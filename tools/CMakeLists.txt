include_directories( AFTER "../src" )

file( GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp )

foreach( source ${srcs} )
	get_filename_component( name ${source} NAME_WE )
	add_executable( ${name} ${source} )
	target_link_libraries( ${name} PUBLIC ${OCR_TRAINING_LIB} OCR_TRAIN )
	target_include_directories( ${name} PUBLIC ${OCR_TRAINING_INCLUDE} )
	target_compile_definitions( ${name} PUBLIC ${OCR_TRAINING_DEFINITION} )
	INSTALL( TARGETS ${name}
			RUNTIME DESTINATION bin
			LIBRARY DESTINATION staticlib
			ARCHIVE DESTINATION lib )
endforeach( source )

