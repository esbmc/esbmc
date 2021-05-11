include(CheckCXXSourceRuns)

# Function to check STP's version
function(check_stp_version stp_include stp_lib)
  # Get lib path
  get_filename_component(stp_lib_path ${stp_lib} PATH)

  try_run(
    STP_RETURNCODE STP_COMPILED ${CMAKE_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/scripts/cmake/try_stp.cpp
    COMPILE_DEFINITIONS -I"${stp_include}" LINK_LIBRARIES -L${stp_lib_path}
                        ${stp_lib}
    RUN_OUTPUT_VARIABLE SRC_OUTPUT)

  if(NOT STP_COMPILED)
    message(
      FATAL_ERROR
        "STP lib found in ${stp_lib_path} but test compilation failed: ${SRC_OUTPUT}"
    )
  endif()

  string(REGEX MATCH "([0-9]*\\.[0-9]*\\.[0-9]*)" stp_version "${SRC_OUTPUT}")
  set(STP_VERSION_STRING
      ${stp_version}
      PARENT_SCOPE)
endfunction(check_stp_version)

# Looking for STP in CAMADA_STP_INCLUDE_DIR
find_path(
  CAMADA_STP_INCLUDE_DIR c_interface.h
  HINTS ${CAMADA_STP_DIR} $ENV{HOME}/stp
  PATH_SUFFIXES include/stp)

find_library(
  CAMADA_STP_LIB stp
  HINTS ${CAMADA_STP_DIR} $ENV{HOME}/stp
  PATH_SUFFIXES lib bin)

# Try to check it dynamically, by compiling a small program that prints STP's
# version
if(CAMADA_STP_INCLUDE_DIR AND CAMADA_STP_LIB)
  # We do not have the STP binary to query for a version. Try to use a small C++
  # program to detect it via the STP_get_version() API call.
  check_stp_version(${CAMADA_STP_INCLUDE_DIR} ${CAMADA_STP_LIB})
endif()

# Alright, now create a list with STP and it's dependencies
find_library(gmp gmp)
list(APPEND CAMADA_STP_LIB "${gmp}")

# handle the QUIETLY and REQUIRED arguments and set STP_FOUND to TRUE if all
# listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  STP
  REQUIRED_VARS CAMADA_STP_LIB CAMADA_STP_INCLUDE_DIR
  VERSION_VAR STP_VERSION_STRING)

mark_as_advanced(CAMADA_STP_LIB CAMADA_STP_INCLUDE_DIR)
