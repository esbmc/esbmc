# Module to setup Apple specific flags

if (APPLE)
    message(STATUS "Detected APPLE")
    # Suppress superfluous warnings about "*.a" having no symbols on macOS X
    set(CMAKE_C_ARCHIVE_CREATE   "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_C_ARCHIVE_FINISH   "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")
    set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")

    # Note: This should check for the CMake version instead of OS
    cmake_policy(SET CMP0079 NEW)

    # MacOS Z3 does not need libgomp
    set(LIBGOMP_LIB "")
    set(OS_FLEX_SMTLIB_FLAGS "")
    set(OS_X86_INCLUDE_FOLDER "/usr/include/${CMAKE_LIBRARY_ARCHITECTURE}")
    set(OS_C2GOTO_FLAGS "")
    set(OS_Z3_LIBS "stdc++ -pthread")
endif()
