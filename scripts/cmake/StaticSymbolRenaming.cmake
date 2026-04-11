include_guard(GLOBAL)

# Rename symbols matching MATCH_REGEXES in a static library, prefixing them with PREFIX.
# Sets OUTPUT_PATH (variable name) to the renamed copy if any symbols matched,
# or to the original library path if nothing needed renaming.
function(esbmc_rename_static_symbols)
  cmake_parse_arguments(ARG "" "STATIC_LIBRARY;OUTPUT_PATH;PREFIX" "MATCH_REGEXES" ${ARGN})

  # Prefer the LLVM toolchain already found by find_package(LLVM), fall back to PATH.
  # Also try versioned names (e.g. llvm-objcopy-17) common on Debian/Ubuntu.
  set(_llvm_tool_hints)
  if(DEFINED LLVM_TOOLS_BINARY_DIR)
    list(APPEND _llvm_tool_hints "${LLVM_TOOLS_BINARY_DIR}")
  endif()

  set(_llvm_version_suffix)
  if(DEFINED LLVM_VERSION_MAJOR)
    set(_llvm_version_suffix "-${LLVM_VERSION_MAJOR}")
  endif()

  find_program(_llvm_nm NAMES "llvm-nm${_llvm_version_suffix}" llvm-nm HINTS ${_llvm_tool_hints} REQUIRED)
  find_program(_llvm_objcopy NAMES "llvm-objcopy${_llvm_version_suffix}" llvm-objcopy HINTS ${_llvm_tool_hints} REQUIRED)
  find_program(_llvm_ranlib NAMES "llvm-ranlib${_llvm_version_suffix}" llvm-ranlib HINTS ${_llvm_tool_hints} REQUIRED)

  get_filename_component(_basename "${ARG_STATIC_LIBRARY}" NAME)
  set(_dir "${CMAKE_CURRENT_BINARY_DIR}/renamed")
  file(MAKE_DIRECTORY "${_dir}")
  set(_renamed "${_dir}/${_basename}")
  set(_syms "${_dir}/${_basename}.syms")

  execute_process(
    COMMAND "${_llvm_nm}" --defined-only -g -j "${ARG_STATIC_LIBRARY}"
    OUTPUT_VARIABLE _nm_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _nm_result)

  if(NOT _nm_result EQUAL 0 OR _nm_output STREQUAL "")
    set(${ARG_OUTPUT_PATH} "${ARG_STATIC_LIBRARY}" PARENT_SCOPE)
    return()
  endif()

  string(REPLACE "\n" ";" _symbols "${_nm_output}")
  set(_map_content "")
  foreach(_sym IN LISTS _symbols)
    string(STRIP "${_sym}" _sym)
    foreach(_regex IN LISTS ARG_MATCH_REGEXES)
      if(_sym MATCHES "${_regex}")
        string(APPEND _map_content "${_sym} ${ARG_PREFIX}${_sym}\n")
        break()
      endif()
    endforeach()
  endforeach()

  if(_map_content STREQUAL "")
    set(${ARG_OUTPUT_PATH} "${ARG_STATIC_LIBRARY}" PARENT_SCOPE)
    return()
  endif()

  file(WRITE "${_syms}" "${_map_content}")

  execute_process(
    COMMAND "${_llvm_objcopy}" "--redefine-syms=${_syms}"
      "${ARG_STATIC_LIBRARY}" "${_renamed}"
    RESULT_VARIABLE _objcopy_result)

  execute_process(
    COMMAND "${_llvm_ranlib}" "${_renamed}"
    RESULT_VARIABLE _ranlib_result)

  if(NOT _objcopy_result EQUAL 0 OR NOT _ranlib_result EQUAL 0)
    message(FATAL_ERROR "Failed to rename symbols in '${ARG_STATIC_LIBRARY}'")
  endif()

  file(STRINGS "${_syms}" _syms_lines)
  list(LENGTH _syms_lines _count)
  message(STATUS "Renamed ${_count} symbols in ${_basename}")
  set(${ARG_OUTPUT_PATH} "${_renamed}" PARENT_SCOPE)
endfunction()
