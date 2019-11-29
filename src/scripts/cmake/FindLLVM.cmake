# Module to find LLVM and checks it's version

# Are we given an LLVM_DIRECTORY?
set (HAVE_LLVM "NOTFOUND")
if (DEFINED LLVM_DIRECTORY)
  include ("${LLVM_DIRECTORY}/lib/cmake/llvm/LLVMConfig.cmake" OPTIONAL RESULT_VARIABLE HAVE_LLVM)
  if (NOT ("${HAVE_LLVM}" STREQUAL "NOTFOUND"))
    include ("${LLVM_DIRECTORY}/lib/cmake/clang/ClangConfig.cmake")
  else()
    message(SEND_ERROR "Could not load LLVMConfig.cmake from requested LLVM directory: ${LLVM_DIRECTORY}")
  endif()

  if (${LLVM_VERSION_MAJOR} GREATER_EQUAL 7)
    message(STATUS "Found valid LLVM. Version: ${LLVM_VERSION}")
  else()
    message(SEND_ERROR "ESBMC needs LLVM 7.0 or greater!")
  endif ()
else()
  message(SEND_ERROR "Could not find LLVM/Clang 7.0 at all: please specify with -DLLVM_DIRECTORY")
endif()
# BUG: For some reason, ESBMC is not linking with Systems LLVM
