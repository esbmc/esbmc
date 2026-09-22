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
