# Module to setup Windows specific flags

if (WIN32)
  # Prebuilt LLVM for Windows doesn't come with CMake files
  message(STATUS "Detected MS Windows")
  # Note: This should check for the CMake version instead of OS
  cmake_policy(SET CMP0079 NEW)  
  
  set(LIBGOMP_LIB "-lgomp -ldl")
  set(OS_FLEX_ANSI_FLAGS "--nounistd")
  set(OS_FLEX_SMTLIB_FLAGS "--wincompat")
  set(OS_X86_INCLUDE_FOLDER "C:/")
  set(OS_C2GOTO_FLAGS "-D_MSVC")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # There are a LOT of warnings from clang headers
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-everything")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-everything")
  set(OS_Z3_LIBS "stdc++")
  
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(OS_Z3_LIBS "")
else()
  message(AUTHOR_WARNING "${CMAKE_CXX_COMPILER_ID} is not tested in Windows. You may run into issues.")	
endif()