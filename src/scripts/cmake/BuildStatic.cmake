# Module to configure static build

message(STATUS "ESBMC will be built in static mode")
set(CMAKE_EXE_LINKER_FLAGS " -static -static-libgcc -static-libstdc++ ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static -lstdc++")
set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS} -static")
set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_STATIC_RUNTIME     ON)
set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})