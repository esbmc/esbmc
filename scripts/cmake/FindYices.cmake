include(CheckCXXSourceRuns)

# Function to check YICES's version
function(check_yices_version yices_include yices_lib)
  # Get lib path
  get_filename_component(yices_lib_path ${yices_lib} PATH)

  try_run(
    YICES_RETURNCODE YICES_COMPILED ${CMAKE_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/scripts/cmake/try_yices.cpp
    COMPILE_DEFINITIONS -I"${yices_include}" LINK_LIBRARIES -L${yices_lib_path}
                        ${yices_lib}
    RUN_OUTPUT_VARIABLE SRC_OUTPUT)

  if(NOT YICES_COMPILED)
    message(
      FATAL_ERROR
        "Yices lib found in ${yices_lib_path} but test compilation failed: ${SRC_OUTPUT}"
    )
  endif()

  string(REGEX MATCH "([0-9]*\\.[0-9]*\\.[0-9]*)" yices_version "${SRC_OUTPUT}")
  set(YICES_VERSION_STRING
      ${yices_version}
      PARENT_SCOPE)
endfunction(check_yices_version)

# Looking for YICES in CAMADA_YICES_INCLUDE_DIR
find_path(
  CAMADA_YICES_INCLUDE_DIR yices.h
  HINTS ${CAMADA_YICES_DIR} $ENV{HOME}/yices
  PATH_SUFFIXES include)

find_library(
  CAMADA_YICES_LIB yices
  HINTS ${CAMADA_YICES_DIR} $ENV{HOME}/yices
  PATH_SUFFIXES lib bin)

# Try to check it dynamically, by compiling a small program that prints YICES's
# version
if(CAMADA_YICES_INCLUDE_DIR AND CAMADA_YICES_LIB)
  # We do not have the YICES binary to query for a version. Try to use a small
  # C++ program to detect it via the YICES_get_version() API call.
  check_yices_version(${CAMADA_YICES_INCLUDE_DIR} ${CAMADA_YICES_LIB})
endif()

# Hack needed for Ubuntu, since it is not linking with static libs from system
if(DEFINED GMP_DIR)
  find_library(
    LIBGMP_CUSTOM gmp
    NAMES libgmp.a
    PATHS ${GMP_DIR}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH)
  message(STATUS "Custom gmp for yices found: ${LIBGMP_CUSTOM}")
  list(APPEND CAMADA_YICES_LIB "${LIBGMP_CUSTOM}")
endif()

# handle the QUIETLY and REQUIRED arguments and set YICES_FOUND to TRUE if all
# listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Yices
  REQUIRED_VARS CAMADA_YICES_LIB CAMADA_YICES_INCLUDE_DIR
  VERSION_VAR YICES_VERSION_STRING)

mark_as_advanced(CAMADA_YICES_LIB CAMADA_YICES_INCLUDE_DIR)
