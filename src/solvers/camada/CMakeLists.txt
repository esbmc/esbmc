# Camada is the SMT abstraction layer this project uses for Z3, CVC5,
# Bitwuzla, MathSAT, and Yices. It can be supplied via a system install or
# downloaded and built on demand. This file:
#   1. Returns early if no camada-backed solver is enabled.
#   2. Looks for an existing camada install (CAMADA_DIR or system).
#   3. Falls back to downloading, building, and installing camada into the
#      build tree when DOWNLOAD_DEPENDENCIES is on.
#   4. Wires camada into the solvercamada library and the global solver list.

if(NOT (ENABLE_Z3 OR ENABLE_CVC5 OR ENABLE_MATHSAT OR ENABLE_YICES OR ENABLE_BITWUZLA))
  return()
endif()

set(CAMADA_DIR "$ENV{HOME}/camada" CACHE PATH "Camada installation prefix")
set(CAMADA_EXTRA_INCLUDE_DIRS)
set(CAMADA_EXTRA_LIBRARY_DIRS)

# Register the include/lib paths for an additional camada dependency tree
# (typically the deps/install/ dir produced by camada's own build) and pull
# in cvc5/STP CMake configs from there if present.
macro(camada_register_deps_tree deps_install_dir)
  if(EXISTS "${deps_install_dir}")
    list(APPEND CAMADA_EXTRA_INCLUDE_DIRS "${deps_install_dir}/include")
    list(APPEND CAMADA_EXTRA_LIBRARY_DIRS
      "${deps_install_dir}/lib"
      "${deps_install_dir}/bin")
    if(CMAKE_LIBRARY_ARCHITECTURE)
      list(APPEND CAMADA_EXTRA_LIBRARY_DIRS
        "${deps_install_dir}/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
    endif()
    list(PREPEND CMAKE_PREFIX_PATH "${deps_install_dir}")

    if(EXISTS "${deps_install_dir}/lib/cmake/cvc5/cvc5Config.cmake"
       AND NOT TARGET cvc5::cvc5)
      find_package(cvc5 CONFIG REQUIRED PATHS "${deps_install_dir}" NO_DEFAULT_PATH)
    endif()
    if(EXISTS "${deps_install_dir}/lib/cmake/STP/STPConfig.cmake"
       AND NOT TARGET stp)
      find_package(STP CONFIG REQUIRED PATHS "${deps_install_dir}" NO_DEFAULT_PATH)
    endif()
  endif()
endmacro()

# Camada's exported camada::camada target lists some shared library paths
# (notably libz3.so) directly in INTERFACE_LINK_LIBRARIES. Under BUILD_STATIC
# we rewrite those to the matching .a if it exists in the same directory so
# the final esbmc binary can be linked statically.
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

# On a re-configure the camada build tree from the previous run is still
# around; re-register its deps so cvc5/STP imported targets are available
# before the find_package(camada ...) call below.
if(DEFINED CAMADA_DOWNLOAD_DIR)
  camada_register_deps_tree("${CAMADA_DOWNLOAD_DIR}/build/deps/install")
endif()

find_package(camada CONFIG QUIET HINTS ${CAMADA_DIR} PATH_SUFFIXES lib/cmake/camada)

#-----------------------------------------------------------------------------#
# Download, build, and install camada when no existing install is found.
#-----------------------------------------------------------------------------#

if(NOT camada_FOUND AND DOWNLOAD_DEPENDENCIES)
  set(CAMADA_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/camada"
      CACHE PATH "Directory used to build downloaded Camada")
  set(CAMADA_ARCHIVE     "${CMAKE_BINARY_DIR}/${ESBMC_CAMADA_NAME}")
  set(CAMADA_SOURCE_DIR  "${CAMADA_DOWNLOAD_DIR}/src")
  set(CAMADA_BUILD_DIR   "${CAMADA_DOWNLOAD_DIR}/build")
  set(CAMADA_INSTALL_DIR "${CAMADA_DOWNLOAD_DIR}/install")
  set(CAMADA_DEPS_INSTALL_DIR "${CAMADA_BUILD_DIR}/deps/install")

  camada_register_deps_tree("${CAMADA_DEPS_INSTALL_DIR}")

  # Download the archive if absent or empty (a previous interrupted run).
  if(EXISTS "${CAMADA_ARCHIVE}")
    file(SIZE "${CAMADA_ARCHIVE}" _camada_archive_size)
    if(_camada_archive_size EQUAL 0)
      file(REMOVE "${CAMADA_ARCHIVE}")
    endif()
  endif()
  if(NOT EXISTS "${CAMADA_ARCHIVE}")
    message(STATUS "Downloading Camada from ${ESBMC_CAMADA_URL}")
    file(DOWNLOAD "${ESBMC_CAMADA_URL}" "${CAMADA_ARCHIVE}"
         SHOW_PROGRESS STATUS _camada_dl_status)
    list(GET _camada_dl_status 0 _camada_dl_result)
    list(GET _camada_dl_status 1 _camada_dl_message)
    if(NOT _camada_dl_result EQUAL 0)
      message(FATAL_ERROR "Failed to download Camada: ${_camada_dl_message}")
    endif()
    file(SIZE "${CAMADA_ARCHIVE}" _camada_archive_size)
    if(_camada_archive_size EQUAL 0)
      message(FATAL_ERROR "Downloaded Camada archive is empty: ${CAMADA_ARCHIVE}")
    endif()
  endif()

  # Treat an existing install as up-to-date unless we need a static rebuild.
  unset(CAMADA_CONFIG_DIR CACHE)
  find_path(CAMADA_CONFIG_DIR
    NAMES camadaConfig.cmake camada-config.cmake
    HINTS "${CAMADA_INSTALL_DIR}"
    PATH_SUFFIXES lib/cmake/camada share/cmake/camada cmake
    NO_DEFAULT_PATH)
  if(BUILD_STATIC AND CAMADA_CONFIG_DIR
     AND NOT EXISTS "${CAMADA_INSTALL_DIR}/lib/libcamada.a")
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

    # GitHub tarballs extract to a single ${repo}-${tag} directory.
    file(GLOB _camada_extracted LIST_DIRECTORIES true "${CAMADA_SOURCE_DIR}/*")
    list(LENGTH _camada_extracted _camada_extracted_count)
    if(_camada_extracted_count EQUAL 1)
      list(GET _camada_extracted 0 CAMADA_PROJECT_SOURCE_DIR)
    else()
      set(CAMADA_PROJECT_SOURCE_DIR "${CAMADA_SOURCE_DIR}")
    endif()

    # Camada has no MSVC-compatible builds for Bitwuzla, CVC5, MathSAT (per
    # camada's CamadaDependencies.cmake) or Yices (autoreconf-only build).
    # Disable them on Windows targets regardless of the user's ENABLE_*.
    set(_camada_unix TRUE)
    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
      set(_camada_unix FALSE)
    endif()
    set(_camada_args)
    foreach(_solver Z3 CVC5 MATHSAT YICES BITWUZLA)
      set(_enabled OFF)
      if(ENABLE_${_solver} AND (_solver STREQUAL "Z3" OR _camada_unix))
        set(_enabled ON)
      endif()
      list(APPEND _camada_args "-DCAMADA_SOLVER_${_solver}_ENABLE=${_enabled}")
    endforeach()
    list(APPEND _camada_args "-DCAMADA_SOLVER_STP_ENABLE=OFF")

    # Multi-config generators (Visual Studio, Xcode) ignore CMAKE_BUILD_TYPE
    # at configure time and pick the configuration via --config later. Pick a
    # default so --config is never empty.
    set(_camada_build_type "${CMAKE_BUILD_TYPE}")
    if(NOT _camada_build_type)
      set(_camada_build_type RelWithDebInfo)
    endif()

    message(STATUS "Configuring Camada")
    execute_process(
      COMMAND ${CMAKE_COMMAND}
        -S "${CAMADA_PROJECT_SOURCE_DIR}"
        -B "${CAMADA_BUILD_DIR}"
        -DCMAKE_BUILD_TYPE=${_camada_build_type}
        -DCMAKE_INSTALL_PREFIX=${CAMADA_INSTALL_DIR}
        -DBUILD_SHARED_LIBS=OFF
        -DCAMADA_DOWNLOAD_DEPENDENCIES=ALL
        -DCAMADA_ENABLE_REGRESSION=OFF
        ${_camada_args}
      RESULT_VARIABLE _camada_status)
    if(NOT _camada_status EQUAL 0)
      message(FATAL_ERROR "Failed to configure Camada")
    endif()

    message(STATUS "Building Camada")
    execute_process(
      COMMAND ${CMAKE_COMMAND} --build "${CAMADA_BUILD_DIR}"
              --config ${_camada_build_type} --parallel
      RESULT_VARIABLE _camada_status)
    if(NOT _camada_status EQUAL 0)
      message(FATAL_ERROR "Failed to build Camada")
    endif()

    message(STATUS "Installing Camada")
    execute_process(
      COMMAND ${CMAKE_COMMAND} --install "${CAMADA_BUILD_DIR}"
              --config ${_camada_build_type}
      RESULT_VARIABLE _camada_status)
    if(NOT _camada_status EQUAL 0)
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
  camada_register_deps_tree("${CAMADA_DEPS_INSTALL_DIR}")
  find_package(camada CONFIG REQUIRED PATHS "${CAMADA_CONFIG_DIR}" NO_DEFAULT_PATH)
  set(CAMADA_DIR "${CAMADA_INSTALL_DIR}" CACHE PATH "Camada installation prefix" FORCE)
endif()

if(NOT camada_FOUND)
  find_package(camada CONFIG REQUIRED HINTS ${CAMADA_DIR} PATH_SUFFIXES lib/cmake/camada)
endif()

if(BUILD_STATIC)
  camada_prefer_static_link_interface(camada::camada)
endif()

foreach(_feature CAMADA_HAVE_Z3 CAMADA_HAVE_CVC5 CAMADA_HAVE_MATHSAT
                 CAMADA_HAVE_YICES CAMADA_HAVE_BITWUZLA)
  if(NOT DEFINED ${_feature})
    message(FATAL_ERROR "Camada package did not export ${_feature}")
  endif()
endforeach()

#-----------------------------------------------------------------------------#
# Build the ESBMC solvercamada bridge library and register it.
#-----------------------------------------------------------------------------#

add_library(solvercamada camada_conv.cpp)

# Camada and its dependencies (cvc5, etc.) are consumed via IMPORTED targets,
# whose include directories CMake propagates as SYSTEM by default. That
# downgrades them to -isystem, letting earlier -I paths (e.g. Homebrew's
# /opt/homebrew/include) shadow Camada's vendored z3++.h with an incompatible
# system copy. Disable SYSTEM-from-imported on the consumer so the includes
# emit as -I and the BEFORE PRIVATE ordering below wins the search race.
set_target_properties(solvercamada PROPERTIES NO_SYSTEM_FROM_IMPORTED TRUE)
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

# Advertise each enabled solver to the rest of ESBMC.
set(_camada_solvers "${ESBMC_AVAILABLE_SOLVERS}")
foreach(_solver z3 cvc5 mathsat yices bitwuzla)
  string(TOUPPER ${_solver} _SOLVER)
  if(ENABLE_${_SOLVER} AND CAMADA_HAVE_${_SOLVER})
    set(ESBMC_ENABLE_${_solver} 1 PARENT_SCOPE)
    list(APPEND _camada_solvers ${_solver})
  endif()
endforeach()
list(REMOVE_DUPLICATES _camada_solvers)
list(JOIN _camada_solvers " " _camada_solvers_str)
set(ESBMC_AVAILABLE_SOLVERS "${_camada_solvers_str}" PARENT_SCOPE)
