set(CAMADA_BACKEND_ENABLED OFF)
if(ENABLE_Z3 OR ENABLE_CVC5 OR ENABLE_MATHSAT OR ENABLE_YICES OR ENABLE_BITWUZLA)
  set(CAMADA_BACKEND_ENABLED ON)
endif()

if(NOT CAMADA_BACKEND_ENABLED)
  return()
endif()

set(CAMADA_DIR "$ENV{HOME}/camada" CACHE PATH "Camada installation prefix")

find_package(camada CONFIG QUIET
  HINTS ${CAMADA_DIR}
  PATH_SUFFIXES lib/cmake/camada)

if(NOT camada_FOUND AND DOWNLOAD_DEPENDENCIES)
  set(CAMADA_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/camada" CACHE PATH "Directory used to extract downloaded Camada")
  set(CAMADA_ARCHIVE "${CMAKE_BINARY_DIR}/${ESBMC_CAMADA_NAME}")

  if(EXISTS "${CAMADA_ARCHIVE}")
    file(SIZE "${CAMADA_ARCHIVE}" CAMADA_ARCHIVE_SIZE)
    if(CAMADA_ARCHIVE_SIZE EQUAL 0)
      file(REMOVE "${CAMADA_ARCHIVE}")
    endif()
  endif()

  if(NOT EXISTS "${CAMADA_ARCHIVE}")
    message(STATUS "Downloading Camada from ${ESBMC_CAMADA_URL}")
    file(DOWNLOAD "${ESBMC_CAMADA_URL}" "${CAMADA_ARCHIVE}"
      SHOW_PROGRESS
      STATUS CAMADA_DOWNLOAD_STATUS)
    list(GET CAMADA_DOWNLOAD_STATUS 0 CAMADA_DOWNLOAD_RESULT)
    list(GET CAMADA_DOWNLOAD_STATUS 1 CAMADA_DOWNLOAD_MESSAGE)
    if(NOT CAMADA_DOWNLOAD_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to download Camada: ${CAMADA_DOWNLOAD_MESSAGE}")
    endif()

    file(SIZE "${CAMADA_ARCHIVE}" CAMADA_ARCHIVE_SIZE)
    if(CAMADA_ARCHIVE_SIZE EQUAL 0)
      message(FATAL_ERROR "Downloaded Camada archive is empty: ${CAMADA_ARCHIVE}")
    endif()
  endif()

  find_path(CAMADA_CONFIG_DIR
    NAMES camadaConfig.cmake camada-config.cmake
    HINTS "${CAMADA_DOWNLOAD_DIR}"
    PATH_SUFFIXES lib/cmake/camada share/cmake/camada cmake
    NO_DEFAULT_PATH)

  if(NOT CAMADA_CONFIG_DIR)
    message(STATUS "Extracting Camada")
    file(REMOVE_RECURSE "${CAMADA_DOWNLOAD_DIR}")
    file(MAKE_DIRECTORY "${CAMADA_DOWNLOAD_DIR}")
    file(ARCHIVE_EXTRACT INPUT "${CAMADA_ARCHIVE}" DESTINATION "${CAMADA_DOWNLOAD_DIR}")

    file(GLOB CAMADA_EXTRACTED_DIRS LIST_DIRECTORIES true "${CAMADA_DOWNLOAD_DIR}/*")
    find_path(CAMADA_CONFIG_DIR
      NAMES camadaConfig.cmake camada-config.cmake
      HINTS "${CAMADA_DOWNLOAD_DIR}" ${CAMADA_EXTRACTED_DIRS}
      PATH_SUFFIXES lib/cmake/camada share/cmake/camada cmake
      NO_DEFAULT_PATH)
  endif()

  if(NOT CAMADA_CONFIG_DIR)
    message(FATAL_ERROR "Downloaded Camada archive does not contain camadaConfig.cmake")
  endif()

  unset(camada_DIR CACHE)
  set(camada_DIR "${CAMADA_CONFIG_DIR}" CACHE PATH "Path to camadaConfig.cmake" FORCE)
  find_package(camada CONFIG REQUIRED
    PATHS "${CAMADA_CONFIG_DIR}"
    NO_DEFAULT_PATH)

  set(CAMADA_DIR "${CAMADA_DOWNLOAD_DIR}" CACHE PATH "Camada installation prefix" FORCE)
endif()

if(NOT camada_FOUND)
  find_package(camada CONFIG REQUIRED
    HINTS ${CAMADA_DIR}
    PATH_SUFFIXES lib/cmake/camada)
endif()

foreach(feature CAMADA_HAVE_Z3 CAMADA_HAVE_CVC5 CAMADA_HAVE_MATHSAT CAMADA_HAVE_YICES CAMADA_HAVE_BITWUZLA)
  if(NOT DEFINED ${feature})
    message(FATAL_ERROR "Camada package did not export ${feature}")
  endif()
endforeach()

add_library(solvercamada camada_conv.cpp)
target_include_directories(solvercamada
  PRIVATE ${Boost_INCLUDE_DIRS}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(solvercamada PUBLIC fmt::fmt camada::camada)

target_link_libraries(solvers INTERFACE solvercamada)

set(CAMADA_ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS}")

if(ENABLE_Z3 AND CAMADA_HAVE_Z3)
  set(ESBMC_ENABLE_z3 1 PARENT_SCOPE)
  list(APPEND CAMADA_ESBMC_AVAILABLE_SOLVERS z3)
endif()

if(ENABLE_CVC5 AND CAMADA_HAVE_CVC5)
  set(ESBMC_ENABLE_cvc5 1 PARENT_SCOPE)
  list(APPEND CAMADA_ESBMC_AVAILABLE_SOLVERS cvc5)
endif()

if(ENABLE_MATHSAT AND CAMADA_HAVE_MATHSAT)
  set(ESBMC_ENABLE_mathsat 1 PARENT_SCOPE)
  list(APPEND CAMADA_ESBMC_AVAILABLE_SOLVERS mathsat)
endif()

if(ENABLE_YICES AND CAMADA_HAVE_YICES)
  set(ESBMC_ENABLE_yices 1 PARENT_SCOPE)
  list(APPEND CAMADA_ESBMC_AVAILABLE_SOLVERS yices)
endif()

if(ENABLE_BITWUZLA AND CAMADA_HAVE_BITWUZLA)
  set(ESBMC_ENABLE_bitwuzla 1 PARENT_SCOPE)
  list(APPEND CAMADA_ESBMC_AVAILABLE_SOLVERS bitwuzla)
endif()

list(REMOVE_DUPLICATES CAMADA_ESBMC_AVAILABLE_SOLVERS)
list(JOIN CAMADA_ESBMC_AVAILABLE_SOLVERS " " CAMADA_ESBMC_AVAILABLE_SOLVERS_STR)
set(ESBMC_AVAILABLE_SOLVERS "${CAMADA_ESBMC_AVAILABLE_SOLVERS_STR}" PARENT_SCOPE)
