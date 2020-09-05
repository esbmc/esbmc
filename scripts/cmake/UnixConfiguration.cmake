# Module to configure non-apple unix systems

if(UNIX AND NOT APPLE)    
    # Linux and BSDs
    message(STATUS "Detected non-apple Unix")
    set(LIBGOMP_LIB "-lgomp -ldl")
endif()