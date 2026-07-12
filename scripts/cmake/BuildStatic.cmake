# Module to configure static build

message(STATUS "ESBMC will be built in static mode")
if(NOT APPLE)
    if(CMAKE_BUILD_TYPE STREQUAL "Sanitizer")
        # Sanitizer runtimes (ASan especially) cannot be fully statically
        # linked: -static drops the dynamic section that libclang_rt.asan
        # needs, producing "undefined reference to _DYNAMIC" at link time.
        # Keep PIC and static dependency archives, but link the executables
        # dynamically so the sanitizer runtime resolves.
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
        set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS} -fPIC")
    else()
        set(CMAKE_EXE_LINKER_FLAGS " -static")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static -fPIC")
        set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS} -static -fPIC")
    endif()
endif()
set(Boost_USE_STATIC_LIBS        ON)
set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
