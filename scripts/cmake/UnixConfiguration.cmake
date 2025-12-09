# Module to configure non-apple unix systems

if(UNIX AND NOT APPLE)    
    # Linux and BSDs
    message(STATUS "Detected non-apple Unix")
    set(LIBGOMP_LIB "-lgomp -ldl")
    set(OS_FLEX_SMTLIB_FLAGS "")
    set(OS_X86_INCLUDE_FOLDER "/usr/include/${CMAKE_LIBRARY_ARCHITECTURE}")
    set(OS_C2GOTO_FLAGS "")
    set(OS_INCLUDE_LIBS m stdc++)
    set(OS_Z3_LIBS "-pthread")
endif()