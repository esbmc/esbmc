# Module to find LLVM and checks it's version

if(NOT (("${LLVM_DIR}" STREQUAL "LLVM_DIR-NOTFOUND") OR ("${LLVM_DIR}" STREQUAL "")))

  find_package(LLVM REQUIRED CONFIG
    PATHS ${LLVM_DIR}
    NO_DEFAULT_PATH
  )

  find_package(Clang REQUIRED CONFIG
    PATHS ${Clang_DIR}
    NO_DEFAULT_PATH
  )
else()
  find_package(LLVM REQUIRED CONFIG)
  find_package(Clang REQUIRED CONFIG)
endif()

if (${LLVM_VERSION_MAJOR} EQUAL 11)
  message(STATUS "LLVM version: ${LLVM_VERSION}")
else()
  message(FATAL_ERROR "Could not find LLVM/Clang 11.0 at all: please specify with -DLLVM_DIR")
endif()
# BUG: For some reason, ESBMC is not linking with Systems LLVM [fb: is this still the case?]

if (CLANG_LINK_CLANG_DYLIB AND NOT BUILD_STATIC)
  set(ESBMC_CLANG_LIBS clang-cpp LLVM)
  message(STATUS "Linking libclang: shared")

  # LLVM upstream decided not to export this variable for the installed tree,
  # since "running 'clang -print-resource-dir'" is good enough:
  # <https://reviews.llvm.org/D49486>
  # Here, we are not running clang, since we only require access to the resources
  # used by the library and do not necessarily want to depend on the clang
  # executable

  try_run(TRY_CLANG_RUNS TRY_CLANG_COMPILES ${CMAKE_CURRENT_BINARY_DIR}
          ${PROJECT_SOURCE_DIR}/scripts/cmake/try_clang.cc
          CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${CLANG_INCLUDE_DIRS}
          RUN_OUTPUT_VARIABLE CLANG_RESOURCE_DIR)

  if (TRY_CLANG_COMPILES AND (TRY_CLANG_RUNS EQUAL 0))
    string(STRIP "${CLANG_RESOURCE_DIR}" CLANG_RESOURCE_DIR)
    set(CLANG_RESOURCE_DIR "${CLANG_INCLUDE_DIRS}/${CLANG_RESOURCE_DIR}")
    message(STATUS "Clang resource dir: ${CLANG_RESOURCE_DIR}")
    file(REAL_PATH "${CLANG_RESOURCE_DIR}/include" ESBMC_CLANG_HEADER_DIR)
    if (IS_DIRECTORY "${ESBMC_CLANG_HEADER_DIR}")
      message("Using system clang's header files in: ${ESBMC_CLANG_HEADER_DIR}")
    else()
      unset(ESBMC_CLANG_HEADER_DIR)
    endif()
  endif()

  if (NOT ESBMC_CLANG_HEADER_DIR)
    message("Not using system clang's header files")
  endif()

else()
  set(ESBMC_CLANG_LIBS clangTooling clangAST clangIndex)
  message(STATUS "Linking libclang: static")
endif()
