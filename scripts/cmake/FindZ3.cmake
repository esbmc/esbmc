include(CheckCXXSourceRuns)

# Function to check Z3's version
function(check_z3_version z3_include z3_lib)
  # Get lib path
  get_filename_component(z3_lib_path ${z3_lib} PATH)

  if(NOT BUILD_SHARED_LIBS)
    find_package(Threads REQUIRED)
    if(Threads_FOUND)
      list(APPEND z3_lib "${CMAKE_THREAD_LIBS_INIT} -ldl")
    endif()
  endif()

  try_run(
    Z3_RETURNCODE Z3_COMPILED ${CMAKE_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/scripts/cmake/try_z3.cpp
    COMPILE_DEFINITIONS -I"${z3_include}" LINK_LIBRARIES -L${z3_lib_path}
                        ${z3_lib}
    RUN_OUTPUT_VARIABLE SRC_OUTPUT)

  if(NOT Z3_COMPILED)
    message(
      FATAL_ERROR
        "Z3 lib found in ${z3_lib_path} but test compilation failed: ${SRC_OUTPUT}"
    )
  endif()

  string(REGEX MATCH "([0-9]*\\.[0-9]*\\.[0-9]*)" z3_version "${SRC_OUTPUT}")
  set(Z3_VERSION_STRING
      ${z3_version}
      PARENT_SCOPE)
endfunction(check_z3_version)

# Looking for Z3 in CAMADA_Z3_INCLUDE_DIR
find_path(CAMADA_Z3_INCLUDE_DIR z3.h HINTS${CAMADA_Z3_DIR} $ENV{HOME}/z3
          PATH_SUFFIXES include)

find_library(CAMADA_Z3_LIB z3 HINTS${CAMADA_Z3_DIR} $ENV{HOME}/z3
             PATH_SUFFIXES lib bin)

# Try to check it dynamically, by compiling a small program that prints Z3's
# version
if(CAMADA_Z3_INCLUDE_DIR AND CAMADA_Z3_LIB)
  # We do not have the Z3 binary to query for a version. Try to use a small C++
  # program to detect it via the Z3_get_version() API call.
  check_z3_version(${CAMADA_Z3_INCLUDE_DIR} ${CAMADA_Z3_LIB})
endif()

# Alright, now create a list with z3 and it's dependencies
if(NOT BUILD_SHARED_LIBS)
  find_package(Threads REQUIRED)
  if(Threads_FOUND)
    list(APPEND CAMADA_Z3_LIB "${CMAKE_THREAD_LIBS_INIT}")
    list(APPEND CAMADA_Z3_LIB "-ldl")
  endif()
endif()

# handle the QUIETLY and REQUIRED arguments and set Z3_FOUND to TRUE if all
# listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Z3
  REQUIRED_VARS CAMADA_Z3_LIB CAMADA_Z3_INCLUDE_DIR
  VERSION_VAR Z3_VERSION_STRING)

mark_as_advanced(CAMADA_Z3_LIB CAMADA_Z3_INCLUDE_DIR)
