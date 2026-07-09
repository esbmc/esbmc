# Wrappers around execute_process() that do not discard the child's exit status.
#
# execute_process() ignores a non-zero exit unless RESULT_VARIABLE is given, so
# a dependency setup script that dies mid-way lets configure continue and the
# real error only surfaces later as an unrelated find_package() failure. See
# the boolector/btor2tools case: a truncated download made setup-btor2tools.sh
# abort, and the reported error was "Could NOT find Btor2Tools".
#
# COMMAND_ERROR_IS_FATAL would do this natively but needs CMake 3.19; this
# project requires 3.18.

include_guard(GLOBAL)

# Abort unless the call site named every argument the wrapper understands.
# Silently dropping a mistyped keyword is the same failure class these
# wrappers exist to remove.
macro(_esbmc_reject_stray_args tag)
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "[${tag}] unexpected arguments: ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR "[${tag}] keywords without values: ${ARG_KEYWORDS_MISSING_VALUES}")
  endif()
endmacro()

# esbmc_execute_or_fail(TAG <tag> WORKING_DIRECTORY <dir> COMMAND <argv>...)
#
# Run COMMAND and abort configure with a FATAL_ERROR naming <tag> if it fails.
function(esbmc_execute_or_fail)
  cmake_parse_arguments(ARG "" "TAG;WORKING_DIRECTORY" "COMMAND" ${ARGN})
  _esbmc_reject_stray_args("${ARG_TAG}")
  string(JOIN " " _pretty ${ARG_COMMAND})

  execute_process(
    COMMAND ${ARG_COMMAND}
    WORKING_DIRECTORY "${ARG_WORKING_DIRECTORY}"
    RESULT_VARIABLE _result)

  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "[${ARG_TAG}] failed (exit ${_result}): ${_pretty}")
  endif()
endfunction()

# esbmc_execute_with_retry(TAG <tag> RETRIES <n> WORKING_DIRECTORY <dir>
#                          COMMAND <argv>...)
#
# As above, but run COMMAND up to <n> times before failing. Intended for steps
# that fetch over the network, where a transient HTTP error would otherwise
# break an otherwise healthy build.
#
# Only use this for a command that is safe to re-run from a partially failed
# state. boolector's contrib/setup-*.sh qualify: download_github() deletes the
# destination directory before unpacking.
function(esbmc_execute_with_retry)
  cmake_parse_arguments(ARG "" "TAG;RETRIES;WORKING_DIRECTORY" "COMMAND" ${ARGN})
  _esbmc_reject_stray_args("${ARG_TAG}")
  string(JOIN " " _pretty ${ARG_COMMAND})

  if(NOT ARG_RETRIES GREATER 0)
    message(FATAL_ERROR "[${ARG_TAG}] RETRIES must be >= 1, got '${ARG_RETRIES}'")
  endif()

  foreach(_attempt RANGE 1 ${ARG_RETRIES})
    execute_process(
      COMMAND ${ARG_COMMAND}
      WORKING_DIRECTORY "${ARG_WORKING_DIRECTORY}"
      RESULT_VARIABLE _result)

    if(_result EQUAL 0)
      return()
    endif()

    message(STATUS
      "[${ARG_TAG}] attempt ${_attempt}/${ARG_RETRIES} failed (exit ${_result})")
  endforeach()

  message(FATAL_ERROR
    "[${ARG_TAG}] failed after ${ARG_RETRIES} attempts: ${_pretty}")
endfunction()
