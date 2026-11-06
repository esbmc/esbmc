include_guard(GLOBAL)

#############################
# Linker selection
#############################

function(_esbmc_try_linker name out_var)
  set(${out_var} FALSE PARENT_SCOPE)
  # -fuse-ld is a driver flag, so probe by actually linking: having the binary
  # on PATH does not mean this compiler driver accepts it.
  set(CMAKE_REQUIRED_LINK_OPTIONS "-fuse-ld=${name}")
  check_cxx_source_compiles("int main(){return 0;}" ESBMC_LINKER_WORKS_${name})
  if(ESBMC_LINKER_WORKS_${name})
    set(${out_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

if(NOT ESBMC_LINKER STREQUAL "default" AND NOT MSVC)
  include(CheckCXXSourceCompiles)

  if(ESBMC_LINKER STREQUAL "auto")
    set(_esbmc_linker_candidates mold lld gold)
    # gold-linked static aarch64 binaries abort in the unwinder on the first
    # throw (uw_init_context_1 -> abort), which the link-only probe cannot see.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
      list(REMOVE_ITEM _esbmc_linker_candidates gold)
    endif()
  else()
    set(_esbmc_linker_candidates ${ESBMC_LINKER})
  endif()

  foreach(_linker IN LISTS _esbmc_linker_candidates)
    _esbmc_try_linker(${_linker} _linker_ok)
    if(_linker_ok)
      add_link_options("-fuse-ld=${_linker}")
      message(STATUS "Using ${_linker} as the linker")
      set(ESBMC_LINKER_IN_USE ${_linker})
      break()
    endif()
  endforeach()

  if(NOT ESBMC_LINKER_IN_USE)
    if(ESBMC_LINKER STREQUAL "auto")
      message(STATUS "No faster linker found; using the toolchain default")
    else()
      message(WARNING "Requested linker '${ESBMC_LINKER}' is unusable; using the toolchain default")
    endif()
  endif()
endif()

#############################
# Duplicate-library warnings
#############################
# Static libraries in a dependency cycle (util_esbmc <-> irep2, ...) are listed
# twice so single-pass linkers resolve them. Apple's ld searches every archive
# anyway and, since Xcode 15, warns once per link about each repeat.
if(APPLE)
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_LINK_OPTIONS "-Wl,-no_warn_duplicate_libraries")
  set(_ld default)
  if(ESBMC_LINKER_IN_USE)
    set(_ld ${ESBMC_LINKER_IN_USE})
    list(APPEND CMAKE_REQUIRED_LINK_OPTIONS "-fuse-ld=${_ld}")
  endif()
  # Keyed on the linker: ld64.lld 18 rejects the flag.
  check_cxx_source_compiles("int main(){return 0;}"
    ESBMC_LINKER_NO_WARN_DUPLICATE_LIBRARIES_${_ld})
  unset(CMAKE_REQUIRED_LINK_OPTIONS)
  if(ESBMC_LINKER_NO_WARN_DUPLICATE_LIBRARIES_${_ld})
    add_link_options("-Wl,-no_warn_duplicate_libraries")
  endif()
endif()

#############################
# Link-time optimization
#############################
if(ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _ipo_supported OUTPUT _ipo_error)
  if(_ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    message(STATUS "Link-time optimization enabled")
  else()
    message(WARNING "ENABLE_IPO is ON but LTO is unsupported here: ${_ipo_error}")
  endif()
endif()

#############################
# Unity builds
#############################
# Off by default: it defeats ccache on incremental edits and can surface
# collisions between file-static names that were previously isolated. Worth it
# for a one-shot full build (CI, release).
if(ENABLE_UNITY_BUILD)
  set(CMAKE_UNITY_BUILD ON)
  set(CMAKE_UNITY_BUILD_BATCH_SIZE 16)
  message(STATUS "Unity builds enabled (batch size ${CMAKE_UNITY_BUILD_BATCH_SIZE})")
endif()
