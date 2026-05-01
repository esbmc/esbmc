set(CAMADA_BACKEND_ENABLED OFF)
if(ENABLE_Z3 OR ENABLE_CVC5 OR ENABLE_MATHSAT OR ENABLE_YICES OR ENABLE_BITWUZLA)
  set(CAMADA_BACKEND_ENABLED ON)
endif()

if(NOT CAMADA_BACKEND_ENABLED)
  return()
endif()

set(CAMADA_DIR "$ENV{HOME}/camada" CACHE PATH "Camada installation prefix")
set(CAMADA_EXTRA_INCLUDE_DIRS)
set(CAMADA_EXTRA_LIBRARY_DIRS)

macro(camada_import_dependency_targets deps_install_dir)
  if(EXISTS "${deps_install_dir}")
    list(APPEND CAMADA_EXTRA_INCLUDE_DIRS
      "${deps_install_dir}/include")
    list(APPEND CAMADA_EXTRA_LIBRARY_DIRS
      "${deps_install_dir}/lib"
      "${deps_install_dir}/bin")
    if(CMAKE_LIBRARY_ARCHITECTURE)
      list(APPEND CAMADA_EXTRA_LIBRARY_DIRS
        "${deps_install_dir}/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
    endif()
    list(PREPEND CMAKE_PREFIX_PATH "${deps_install_dir}")

    if(EXISTS "${deps_install_dir}/lib/cmake/cvc5/cvc5Config.cmake" AND NOT TARGET cvc5::cvc5)
      find_package(cvc5 CONFIG REQUIRED
        PATHS "${deps_install_dir}"
        NO_DEFAULT_PATH)
    endif()

    if(EXISTS "${deps_install_dir}/lib/cmake/STP/STPConfig.cmake" AND NOT TARGET stp)
      find_package(STP CONFIG REQUIRED
        PATHS "${deps_install_dir}"
        NO_DEFAULT_PATH)
    endif()
  endif()
endmacro()

function(camada_prefer_static_link_interface target)
  if(NOT TARGET ${target})
    return()
  endif()

  get_target_property(libs ${target} INTERFACE_LINK_LIBRARIES)
  if(NOT libs)
    return()
  endif()

  set(static_libs)
  foreach(lib IN LISTS libs)
    set(static_lib "${lib}")
    if(IS_ABSOLUTE "${lib}" AND
       "${lib}" MATCHES "\\${CMAKE_SHARED_LIBRARY_SUFFIX}(\\.[0-9]+)*$")
      get_filename_component(lib_dir "${lib}" DIRECTORY)
      get_filename_component(lib_name "${lib}" NAME_WE)
      set(candidate "${lib_dir}/${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")
      if(EXISTS "${candidate}")
        set(static_lib "${candidate}")
      endif()
    endif()
    list(APPEND static_libs "${static_lib}")
  endforeach()

  set_target_properties(${target} PROPERTIES
    INTERFACE_LINK_LIBRARIES "${static_libs}")
endfunction()

if(DEFINED CAMADA_DOWNLOAD_DIR)
  camada_import_dependency_targets("${CAMADA_DOWNLOAD_DIR}/build/deps/install")
endif()

find_package(camada CONFIG QUIET
  HINTS ${CAMADA_DIR}
  PATH_SUFFIXES lib/cmake/camada)

if(NOT camada_FOUND AND DOWNLOAD_DEPENDENCIES)
  set(CAMADA_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/camada" CACHE PATH "Directory used to build downloaded Camada")
  set(CAMADA_ARCHIVE "${CMAKE_BINARY_DIR}/${ESBMC_CAMADA_NAME}")
  set(CAMADA_SOURCE_DIR "${CAMADA_DOWNLOAD_DIR}/src")
  set(CAMADA_BUILD_DIR "${CAMADA_DOWNLOAD_DIR}/build")
  set(CAMADA_INSTALL_DIR "${CAMADA_DOWNLOAD_DIR}/install")
  set(CAMADA_DEPS_INSTALL_DIR "${CAMADA_BUILD_DIR}/deps/install")
  camada_import_dependency_targets("${CAMADA_DEPS_INSTALL_DIR}")
  unset(CAMADA_CONFIG_DIR CACHE)

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
    HINTS "${CAMADA_INSTALL_DIR}"
    PATH_SUFFIXES lib/cmake/camada share/cmake/camada cmake
    NO_DEFAULT_PATH)

  if(BUILD_STATIC AND CAMADA_CONFIG_DIR AND
     NOT EXISTS "${CAMADA_INSTALL_DIR}/lib/libcamada.a")
    message(STATUS "Rebuilding Camada as a static library")
    file(REMOVE_RECURSE "${CAMADA_BUILD_DIR}" "${CAMADA_INSTALL_DIR}")
    unset(CAMADA_CONFIG_DIR CACHE)
    set(CAMADA_CONFIG_DIR)
  endif()

  if(NOT CAMADA_CONFIG_DIR)
    message(STATUS "Extracting Camada")
    file(REMOVE_RECURSE "${CAMADA_SOURCE_DIR}")
    file(MAKE_DIRECTORY "${CAMADA_SOURCE_DIR}")
    file(ARCHIVE_EXTRACT INPUT "${CAMADA_ARCHIVE}" DESTINATION "${CAMADA_SOURCE_DIR}")

    file(GLOB CAMADA_EXTRACTED_DIRS LIST_DIRECTORIES true "${CAMADA_SOURCE_DIR}/*")
    list(LENGTH CAMADA_EXTRACTED_DIRS CAMADA_EXTRACTED_DIR_COUNT)
    if(CAMADA_EXTRACTED_DIR_COUNT EQUAL 1)
      list(GET CAMADA_EXTRACTED_DIRS 0 CAMADA_PROJECT_SOURCE_DIR)
    else()
      set(CAMADA_PROJECT_SOURCE_DIR "${CAMADA_SOURCE_DIR}")
    endif()

    message(STATUS "Configuring Camada")
    execute_process(
      COMMAND ${CMAKE_COMMAND}
        -S "${CAMADA_PROJECT_SOURCE_DIR}"
        -B "${CAMADA_BUILD_DIR}"
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${CAMADA_INSTALL_DIR}
        -DBUILD_SHARED_LIBS=OFF
        -DCAMADA_DOWNLOAD_DEPENDENCIES=ALL
        -DCAMADA_ENABLE_REGRESSION=OFF
        -DCAMADA_SOLVER_BITWUZLA_ENABLE=ON
        -DCAMADA_SOLVER_CVC5_ENABLE=ON
        -DCAMADA_SOLVER_MATHSAT_ENABLE=OFF
        -DCAMADA_SOLVER_STP_ENABLE=OFF
        -DCAMADA_SOLVER_YICES_ENABLE=OFF
        -DCAMADA_SOLVER_Z3_ENABLE=ON
      RESULT_VARIABLE CAMADA_CONFIGURE_RESULT)
    if(NOT CAMADA_CONFIGURE_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to configure Camada")
    endif()

    message(STATUS "Building Camada")
    execute_process(
      COMMAND ${CMAKE_COMMAND} --build "${CAMADA_BUILD_DIR}" --parallel 32
      RESULT_VARIABLE CAMADA_BUILD_RESULT)
    if(NOT CAMADA_BUILD_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to build Camada")
    endif()

    message(STATUS "Installing Camada")
    execute_process(
      COMMAND ${CMAKE_COMMAND} --install "${CAMADA_BUILD_DIR}"
      RESULT_VARIABLE CAMADA_INSTALL_RESULT)
    if(NOT CAMADA_INSTALL_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to install Camada")
    endif()

    find_path(CAMADA_CONFIG_DIR
      NAMES camadaConfig.cmake camada-config.cmake
      HINTS "${CAMADA_INSTALL_DIR}"
      PATH_SUFFIXES lib/cmake/camada share/cmake/camada cmake
      NO_DEFAULT_PATH)
  endif()

  if(NOT CAMADA_CONFIG_DIR)
    message(FATAL_ERROR "Downloaded Camada build did not install camadaConfig.cmake")
  endif()

  unset(camada_DIR CACHE)
  set(camada_DIR "${CAMADA_CONFIG_DIR}" CACHE PATH "Path to camadaConfig.cmake" FORCE)
  camada_import_dependency_targets("${CAMADA_DEPS_INSTALL_DIR}")

  find_package(camada CONFIG REQUIRED
    PATHS "${CAMADA_CONFIG_DIR}"
    NO_DEFAULT_PATH)

  set(CAMADA_DIR "${CAMADA_INSTALL_DIR}" CACHE PATH "Camada installation prefix" FORCE)
endif()

if(NOT camada_FOUND)
  find_package(camada CONFIG REQUIRED
    HINTS ${CAMADA_DIR}
    PATH_SUFFIXES lib/cmake/camada)
endif()

# Camada and its solver dependencies (cvc5, etc.) are consumed as IMPORTED
# targets, whose include dirs CMake propagates as SYSTEM by default. That
# downgrades them to -isystem and lets earlier -I paths (e.g. Homebrew's
# /opt/homebrew/include) shadow Camada's vendored z3++.h with an
# incompatible system copy. Force non-SYSTEM on every imported target that
# Camada brings in so their includes emit as -I.
foreach(_camada_imported_target
        camada::camada cvc5::cvc5 cvc5::cvc5parser stp::stp)
  if(TARGET ${_camada_imported_target})
    set_target_properties(${_camada_imported_target} PROPERTIES
      IMPORTED_NO_SYSTEM TRUE)
  endif()
endforeach()

if(BUILD_STATIC)
  camada_prefer_static_link_interface(camada::camada)
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
if(CAMADA_EXTRA_INCLUDE_DIRS)
  target_include_directories(solvercamada BEFORE PRIVATE ${CAMADA_EXTRA_INCLUDE_DIRS})
endif()
if(CAMADA_EXTRA_LIBRARY_DIRS)
  target_link_directories(solvercamada PUBLIC ${CAMADA_EXTRA_LIBRARY_DIRS})
endif()
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
