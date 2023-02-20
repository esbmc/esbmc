# Module to setup Windows specific flags

if (WIN32)
  # Prebuilt LLVM for Windows doesn't come with CMake files
  message(STATUS "Detected MS Windows")
  # Note: This should check for the CMake version instead of OS
  cmake_policy(SET CMP0079 NEW)

  # std::min fails on windows after including windows.h
  # this is a workaround
  add_compile_definitions(NOMINMAX=1)

  find_package (Python COMPONENTS Interpreter)
  if(NOT Python_FOUND)
	message(WARNING "Python not found, cmake will assume that it is on path")
	set(Python_EXECUTABLE python)
  endif()
  message(STATUS "Found Python: ${Python_EXECUTABLE}")
  set(LIBGOMP_LIB "-lgomp -ldl")
  set(OS_FLEX_SMTLIB_FLAGS "--wincompat")
  set(OS_X86_INCLUDE_FOLDER "C:/")
  set(OS_C2GOTO_FLAGS "-D_MSVC")

  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	  # There are a LOT of warnings from clang headers
	  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-everything")
	  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-everything")
      set(OS_Z3_LIBS "stdc++")
  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      # CMP0092 does not add any warning level by default
      # TODO: This can be removed when we move to CMake >= 3.15
      cmake_policy(SET CMP0092 NEW)
	  add_compile_options(/bigobj)
	  add_compile_definitions(/W1)
	  string(REGEX REPLACE "/W[1-3]" "/W1" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
      set(OS_Z3_LIBS "")
  else()
	  message(AUTHOR_WARNING "${CMAKE_CXX_COMPILER_ID} is not tested in Windows. You may run into issues.")
	endif()

  if(DOWNLOAD_WINDOWS_DEPENDENCIES)
    download_zip_and_extract(LLVM https://gitlab.com/Anthonysdu/llvm11/-/raw/main/llvm+clang+lld-11.0.0-x86_64-windows-msvc-release-mt.zip)
    download_zip_and_extract(Z3 https://github.com/Z3Prover/z3/releases/download/z3-4.11.2/z3-4.11.2-x64-win.zip)
    set(LLVM_DIR ${CMAKE_BINARY_DIR}/LLVM/llvm+clang+lld-11.0.0-x86_64-windows-msvc-release-mt)
    set(Clang_DIR ${CMAKE_BINARY_DIR}/LLVM/llvm+clang+lld-11.0.0-x86_64-windows-msvc-release-mt)
    set(Z3_DIR ${CMAKE_BINARY_DIR}/Z3/z3-4.11.2-x64-win)
  endif()

  # Produce static builds in windows
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  set(Boost_USE_STATIC_LIBS        ON)

endif()
