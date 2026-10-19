# Every backend is off until the camada probe below proves it available; these
# defaults feed solver_config.h.in.
set(ESBMC_ENABLE_z3 0)
set(ESBMC_ENABLE_cvc5 0)
set(ESBMC_ENABLE_mathsat 0)
set(ESBMC_ENABLE_yices 0)
set(ESBMC_ENABLE_bitwuzla 0)
set(ESBMC_ENABLE_smtlib 0)

# Camada is the SMT abstraction layer this project uses for Z3, CVC5,
# Bitwuzla, MathSAT, and Yices. This file:
#   1. Returns early if no camada-backed solver is enabled.
#   2. Uses the install CAMADA_DIR names, or else, when DOWNLOAD_DEPENDENCIES
#      is on, downloads and unpacks the pinned prebuilt release into the build
#      tree (for Debug/DebugOpt, builds that release from source instead);
#      failing both, looks for a system install.
#   3. Wires camada into the solvers library and the global solver list.

if(NOT (ENABLE_Z3 OR ENABLE_CVC5 OR ENABLE_MATHSAT OR ENABLE_YICES OR ENABLE_BITWUZLA))
  return()
endif()

# Empty by default: a camada tree left in a developer's home directory must not
# silently take the place of the pinned release (its version is unchecked and
# its build options unknown). Set -DCAMADA_DIR=... to use one deliberately, e.g.
# for MathSAT or Yices, which the release archives do not carry.
set(CAMADA_DIR "" CACHE PATH "Camada installation prefix")
# Build trees configured when camada was built from source cached its install
# here; drop it so they move to the pinned release.
if(CAMADA_DIR STREQUAL "${CMAKE_BINARY_DIR}/camada/install")
  set(CAMADA_DIR "" CACHE PATH "Camada installation prefix" FORCE)
endif()
set(CAMADA_EXTRA_INCLUDE_DIRS)
set(CAMADA_EXTRA_LIBRARY_DIRS)

# A hand-built camada exports absolute paths to its own dependency tree
# (…/build/deps/install), and names cvc5::cvc5 / stp as imported targets when
# built with those solvers. Register that tree, or find_package(camada) fails
# on the missing targets. CAMADA_DEPS_DIR overrides the guess for an install
# whose build tree is not a sibling of the prefix.
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

set(CAMADA_DEPS_DIR "" CACHE PATH
    "camada's dependency install tree (…/build/deps/install)")
if(CAMADA_DEPS_DIR)
  camada_register_deps_tree("${CAMADA_DEPS_DIR}")
elseif(CAMADA_DIR)
  camada_register_deps_tree("${CAMADA_DIR}/build/deps/install")
  camada_register_deps_tree("${CAMADA_DIR}/../camada/build/deps/install")
endif()

#-----------------------------------------------------------------------------#
# Download and unpack the pinned camada release when no install is given.
#-----------------------------------------------------------------------------#

set(_camada_prefix "${CAMADA_DIR}")
if(NOT CAMADA_DIR AND DOWNLOAD_DEPENDENCIES)
  # Debug and DebugOpt build camada from source in a matching configuration
  # instead of using the release archive: camada's checked handles stay on, so
  # a stale SMTExprRef aborts instead of being undefined behaviour, and on
  # Windows it links against ESBMC's debug CRT (/MTd), which the release-only
  # archive cannot.
  set(_camada_build_type "")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(_camada_build_type Debug)
  elseif(CMAKE_BUILD_TYPE STREQUAL "DebugOpt")
    set(_camada_build_type RelWithDebInfo)
  endif()
  if(_camada_build_type)
    set(_camada_url "${ESBMC_CAMADA_SOURCE_URL}")
    set(_camada_hash "${ESBMC_CAMADA_SOURCE_HASH}")
  else()
    set(_camada_url "${ESBMC_CAMADA_ARCHIVE_URL}")
    set(_camada_hash "${ESBMC_CAMADA_ARCHIVE_HASH}")
  endif()

  # Camada has no MSVC-compatible builds for Bitwuzla, CVC5, MathSAT or Yices,
  # so a source build on Windows carries Z3 only.
  set(_camada_args)
  foreach(_solver Z3 CVC5 MATHSAT YICES BITWUZLA)
    set(_enabled OFF)
    if(ENABLE_${_solver} AND (_solver STREQUAL "Z3" OR NOT WIN32))
      set(_enabled ON)
    endif()
    list(APPEND _camada_args "-DCAMADA_SOLVER_${_solver}_ENABLE=${_enabled}")
  endforeach()
  list(APPEND _camada_args "-DCAMADA_SOLVER_STP_ENABLE=OFF")
  if(MSVC)
    list(APPEND _camada_args
      "-DCMAKE_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}")
  endif()

  # Unpack into a directory named after everything that shapes the result, so a
  # new pin (even a retagged release at the same URL), build type or solver set
  # starts afresh instead of reusing the previous one.
  string(SHA256 _camada_key
         "${_camada_url}${_camada_hash}${_camada_build_type}${_camada_args}")
  string(SUBSTRING "${_camada_key}" 0 16 _camada_key)
  set(_camada_root "${CMAKE_BINARY_DIR}/camada/${_camada_key}")
  set(_camada_prefix "${_camada_root}/install")
  if(_camada_build_type)
    camada_register_deps_tree("${_camada_root}/build/deps/install")
  endif()

  if(NOT EXISTS "${_camada_prefix}/lib/cmake/camada/camadaConfig.cmake")
    get_filename_component(_camada_archive_name "${_camada_url}" NAME)
    set(_camada_archive "${CMAKE_BINARY_DIR}/camada/${_camada_archive_name}")
    message(STATUS "Downloading Camada from ${_camada_url}")
    if(_camada_hash)
      file(DOWNLOAD "${_camada_url}" "${_camada_archive}"
           STATUS _camada_dl_status EXPECTED_HASH SHA256=${_camada_hash})
    else()
      message(WARNING "Camada's pinned hash is empty: downloading it unverified")
      file(DOWNLOAD "${_camada_url}" "${_camada_archive}"
           STATUS _camada_dl_status)
    endif()
    list(GET _camada_dl_status 0 _camada_dl_result)
    if(NOT _camada_dl_result EQUAL 0)
      list(GET _camada_dl_status 1 _camada_dl_message)
      file(REMOVE "${_camada_archive}")
      message(FATAL_ERROR "Failed to download Camada: ${_camada_dl_message}")
    endif()

    file(REMOVE_RECURSE "${_camada_root}")
    if(NOT _camada_build_type)
      file(ARCHIVE_EXTRACT INPUT "${_camada_archive}" DESTINATION "${_camada_prefix}")
    else()
      file(ARCHIVE_EXTRACT INPUT "${_camada_archive}" DESTINATION "${_camada_root}/src")
      # GitHub tarballs extract to a single ${repo}-${tag} directory.
      file(GLOB _camada_src LIST_DIRECTORIES true "${_camada_root}/src/*")
      message(STATUS "Building Camada (${_camada_build_type}) from source")
      execute_process(
        COMMAND ${CMAKE_COMMAND} -S "${_camada_src}" -B "${_camada_root}/build"
          -DCMAKE_BUILD_TYPE=${_camada_build_type}
          -DCMAKE_INSTALL_PREFIX=${_camada_prefix}
          -DBUILD_SHARED_LIBS=OFF
          -DCAMADA_DOWNLOAD_DEPENDENCIES=ALL
          -DCAMADA_ENABLE_REGRESSION=OFF
          ${_camada_args}
        RESULT_VARIABLE _camada_status)
      if(_camada_status EQUAL 0)
        execute_process(
          COMMAND ${CMAKE_COMMAND} --build "${_camada_root}/build"
                  --config ${_camada_build_type} --parallel
          RESULT_VARIABLE _camada_status)
      endif()
      if(_camada_status EQUAL 0)
        execute_process(
          COMMAND ${CMAKE_COMMAND} --install "${_camada_root}/build"
                  --config ${_camada_build_type}
          RESULT_VARIABLE _camada_status)
      endif()
      if(NOT _camada_status EQUAL 0)
        message(FATAL_ERROR "Failed to build Camada from ${_camada_url}")
      endif()
      camada_register_deps_tree("${_camada_root}/build/deps/install")
    endif()
    file(REMOVE "${_camada_archive}")
    if(NOT EXISTS "${_camada_prefix}/lib/cmake/camada/camadaConfig.cmake")
      message(FATAL_ERROR
        "${_camada_url} produced no lib/cmake/camada/camadaConfig.cmake")
    endif()
  endif()
endif()

# camada_DIR is cached by find_package; drop it so a changed CAMADA_DIR or pin
# is searched again rather than the previous configure's install reused.
unset(camada_DIR CACHE)
find_package(camada CONFIG REQUIRED HINTS ${_camada_prefix}
             PATH_SUFFIXES lib/cmake/camada)

# Camada is built for a single configuration (CMAKE_BUILD_TYPE on
# single-config generators, RelWithDebInfo when invoked from a multi-config
# generator like Visual Studio). The exported camada::camada target
# therefore only carries IMPORTED_LOCATION_<one-config>. When ESBMC is
# itself driven by a multi-config generator, CMake validates the imported
# target against every entry in CMAKE_CONFIGURATION_TYPES (Debug, Release,
# MinSizeRel, RelWithDebInfo) and errors out on the missing ones. Map
# every consumer config to the configurations camada actually provides so
# the target resolves uniformly.
if(TARGET camada::camada)
  set(_camada_config_fallbacks RelWithDebInfo Release Debug MinSizeRel "")
  foreach(_cfg DEBUG RELEASE MINSIZEREL RELWITHDEBINFO)
    set_property(TARGET camada::camada PROPERTY
      MAP_IMPORTED_CONFIG_${_cfg} ${_camada_config_fallbacks})
  endforeach()
endif()

# Under BUILD_STATIC the user expects no shared-library dependencies from
# camada's link interface. Pre-v0.9 camada installs (and hand-built ones
# without BUILD_SHARED_LIBS=OFF) export absolute paths to .so/.dylib files
# in INTERFACE_LINK_LIBRARIES. Fail loudly so a stale CAMADA_DIR pin is
# diagnosed at configure time instead of producing a silently dynamically
# linked esbmc.
if(BUILD_STATIC AND TARGET camada::camada)
  get_target_property(_camada_link_iface camada::camada INTERFACE_LINK_LIBRARIES)
  if(_camada_link_iface)
    set(_camada_shared_paths)
    foreach(_lib IN LISTS _camada_link_iface)
      if("${_lib}" MATCHES "\\.(so|dylib)(\\.[0-9]+)*$"
         OR "${_lib}" MATCHES "\\.dll$")
        list(APPEND _camada_shared_paths "${_lib}")
      endif()
    endforeach()
    if(_camada_shared_paths)
      string(REPLACE ";" "\n  " _camada_shared_str "${_camada_shared_paths}")
      message(FATAL_ERROR
        "BUILD_STATIC=ON but camada::camada's INTERFACE_LINK_LIBRARIES "
        "references shared libraries:\n  ${_camada_shared_str}\n"
        "Camada v0.9+ exports static archives when built with "
        "BUILD_SHARED_LIBS=OFF. Either upgrade your CAMADA_DIR install to "
        "v0.9+ or rebuild it with BUILD_SHARED_LIBS=OFF.")
    endif()
  endif()
endif()

foreach(_feature CAMADA_HAVE_Z3 CAMADA_HAVE_CVC5 CAMADA_HAVE_MATHSAT
                 CAMADA_HAVE_YICES CAMADA_HAVE_BITWUZLA CAMADA_HAVE_SMTLIB)
  if(NOT DEFINED ${_feature})
    message(FATAL_ERROR "Camada package did not export ${_feature}")
  endif()
endforeach()

#-----------------------------------------------------------------------------#
# Build the solvers library and register the backends it provides.
#-----------------------------------------------------------------------------#

add_library(solvers
  solve.cpp
  # SMT-backed proof of the simplifier's rewrites, behind a build flag (#6971).
  # Upstream added this under src/solvers/smt/; the flatten puts it here.
  simplification_equivalence.cpp
  # Solver-agnostic core: the memory model, byte ops, casts, overflow and FP
  # encoding every backend shares.
  smt_solver.cpp
  smt_conv.cpp
  smt_byteops.cpp
  smt_casts.cpp
  smt_bitcast.cpp
  smt_memspace.cpp
  smt_overflow.cpp
  # IEEE754 constants/semantics and FP predicates split out of smt_solver.cpp
  smt_fp_conv.cpp
  # Real-arithmetic FP path (--ir --ir-ieee)
  ir_ieee_conv.cpp
  # Address-space bookkeeping the memory model uses
  pointer_logic.cpp
  # The backend: every solver camada supports, plus its SMT-LIB text mode
  camada_conv.cpp
  oneshot_options.cpp
)

# Camada and its dependencies (cvc5, etc.) are consumed via IMPORTED targets,
# whose include directories CMake propagates as SYSTEM by default. That
# downgrades them to -isystem, letting earlier -I paths (e.g. Homebrew's
# /opt/homebrew/include) shadow Camada's vendored z3++.h with an incompatible
# system copy. Disable SYSTEM-from-imported on the consumer so the includes
# emit as -I and the BEFORE PRIVATE ordering below wins the search race.
set_target_properties(solvers PROPERTIES NO_SYSTEM_FROM_IMPORTED TRUE)

target_include_directories(solvers
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${Boost_INCLUDE_DIRS})
if(CAMADA_EXTRA_INCLUDE_DIRS)
  target_include_directories(solvers BEFORE PRIVATE ${CAMADA_EXTRA_INCLUDE_DIRS})
endif()
if(CAMADA_EXTRA_LIBRARY_DIRS)
  target_link_directories(solvers PUBLIC ${CAMADA_EXTRA_LIBRARY_DIRS})
endif()

target_link_libraries(solvers PUBLIC fmt::fmt camada::camada filesystem)

# Advertise each enabled solver to the rest of ESBMC.
set(_camada_solvers "${ESBMC_AVAILABLE_SOLVERS}")
foreach(_solver z3 cvc5 mathsat yices bitwuzla)
  string(TOUPPER ${_solver} _SOLVER)
  if(ENABLE_${_SOLVER} AND CAMADA_HAVE_${_SOLVER})
    set(ESBMC_ENABLE_${_solver} 1)
    list(APPEND _camada_solvers ${_solver})
  elseif(ENABLE_${_SOLVER})
    message(WARNING "ENABLE_${_SOLVER} is on, but this camada has no "
                    "${_solver} backend; set CAMADA_DIR to a camada built with it")
  endif()
endforeach()

# smtlib needs no solver library of its own: it pipes to whatever the user
# names with --smtlib-solver-prog, runs --smtlib-oneshot-prog on a dumped
# script, or just writes the script out. Camada's SMT-LIB backend serves all
# three shapes, but it drives the external process with POSIX
# fork/exec/setrlimit/select, so camada forces it off on Windows -- follow
# CAMADA_HAVE_SMTLIB rather than assuming it exists wherever camada does.
if(ENABLE_SMTLIB AND CAMADA_HAVE_SMTLIB)
  set(ESBMC_ENABLE_smtlib 1)
  list(APPEND _camada_solvers smtlib)
endif()
list(REMOVE_DUPLICATES _camada_solvers)
list(JOIN _camada_solvers " " _camada_solvers_str)
set(ESBMC_AVAILABLE_SOLVERS "${_camada_solvers_str}" PARENT_SCOPE)

# Windows: ship the release's libz3.dll alongside esbmc.exe. Without it,
# esbmc.exe exits silently at startup when the OS loader cannot find the DLL.
# A source build leaves it in camada's dependency tree instead.
foreach(_dll "${_camada_prefix}/bin/libz3.dll"
             "${_camada_root}/build/deps/install/bin/libz3.dll")
  if(WIN32 AND EXISTS "${_dll}")
    install(FILES "${_dll}" DESTINATION bin)
    break()
  endif()
endforeach()

# The release archives link bare -lgmp/-lmpfr. Homebrew on Apple Silicon keeps
# them in /opt/homebrew/lib, which the linker does not search by default.
if(APPLE)
  foreach(_lib gmp mpfr)
    find_library(_camada_${_lib} ${_lib})
    if(_camada_${_lib})
      get_filename_component(_dir "${_camada_${_lib}}" DIRECTORY)
      target_link_directories(solvers PUBLIC "${_dir}")
    endif()
  endforeach()
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/solver_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/solver_config.h")
