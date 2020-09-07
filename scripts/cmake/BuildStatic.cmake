# Module to configure static build

message(STATUS "ESBMC will be built in static mode")
if(UNIX AND NOT APPLE)
    set(CMAKE_EXE_LINKER_FLAGS " -static")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static -fPIC")
    set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS} -static -fPIC")
endif()

if(UNIX)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	  set(MSVC_RUNTIME_LIBRARY MultiThreaded)
endif()
set(Boost_USE_STATIC_LIBS        ON)

