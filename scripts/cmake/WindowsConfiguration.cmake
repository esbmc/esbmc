# Module to setup Windows specific flags

if (WIN32)
  # Prebuilt LLVM for Windows doesn't come with CMake files
  message(STATUS "Detected MS Windows")
  #add_definitions(/bigobj)
  # There are a LOT of warnings from clang headers
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-everything")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-everything")

endif()