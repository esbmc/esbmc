# Module to setup Apple specific flags

if(APPLE)
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
    set(OS_INCLUDE_LIBS "")
    set(OS_Z3_LIBS "stdc++ -pthread")

    # macOS has had no /usr/include since 10.14, so the C library headers are
    # reachable only through an SDK. Both c2goto and every runtime parse
    # (ESBMC_C2GOTO_SYSROOT, baked into ac_config.h) need one; without it the
    # build dies while generating the libc models, on a "'complex.h' file not
    # found" that names neither CMake nor the SDK (esbmc/esbmc#6234).
    if(NOT DEFINED C2GOTO_SYSROOT)
        if(CMAKE_OSX_SYSROOT AND IS_DIRECTORY "${CMAKE_OSX_SYSROOT}")
            set(detected_sdk "${CMAKE_OSX_SYSROOT}")
        else()
            execute_process(COMMAND xcrun --show-sdk-path
                OUTPUT_VARIABLE detected_sdk
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE xcrun_result)
            if(NOT xcrun_result EQUAL 0 OR NOT detected_sdk)
                set(detected_sdk "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
            endif()
        endif()
        set(C2GOTO_SYSROOT "${detected_sdk}" CACHE PATH
            "SDK the C/C++ frontend parses against (auto-detected on macOS)")
    endif()

    if(NOT IS_DIRECTORY "${C2GOTO_SYSROOT}")
        message(FATAL_ERROR
            "C2GOTO_SYSROOT is '${C2GOTO_SYSROOT}', which is not a directory. "
            "ESBMC needs a macOS SDK to find the system headers. Install the "
            "command line tools with 'xcode-select --install', or pass "
            "-DC2GOTO_SYSROOT=<path to MacOSX.sdk>.")
    endif()
    message(STATUS "C2GOTO_SYSROOT: ${C2GOTO_SYSROOT}")
endif()
