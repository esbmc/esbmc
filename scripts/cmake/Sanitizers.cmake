# Module to set flags for sanitizers

# Inspired on http://www.stablecoder.ca/2018/02/01/analyzer-build-types.html

# Instrumentation only: -O and -g belong to CMAKE_BUILD_TYPE, so that any build
# type can be sanitized rather than only the dedicated one (esbmc/esbmc#1380).
# -fsanitize-recover=address makes findings recoverable so that, combined with
# halt_on_error=0, the process continues past a report instead of aborting. That
# is required for the build-time c2goto invocations to survive bundled-clang
# findings (see sanitizers.yml) and lets one esbmc run surface several findings.
set(SANITIZER_FLAGS_address
    "-fsanitize=address -fsanitize-recover=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer")
set(SANITIZER_FLAGS_thread "-fsanitize=thread")
set(SANITIZER_FLAGS_leak "-fsanitize=leak -fno-omit-frame-pointer")
set(SANITIZER_FLAGS_memory
    "-fsanitize=memory -fno-optimize-sibling-calls -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer")
set(SANITIZER_FLAGS_undefined "-fsanitize=undefined -fno-sanitize=vptr")

set(SANITIZER_KNOWN address thread leak memory undefined)

# The clang -fsanitize= spellings are the canonical ones; the *SAN names are
# kept because SANITIZER_TYPE and `build.sh -s` have always used them.
set(SANITIZER_ALIAS_asan address)
set(SANITIZER_ALIAS_tsan thread)
set(SANITIZER_ALIAS_lsan leak)
set(SANITIZER_ALIAS_msan memory)
set(SANITIZER_ALIAS_ubsan undefined)

set(ENABLE_SANITIZERS "" CACHE STRING
    "Sanitizers to instrument with, independent of CMAKE_BUILD_TYPE; comma- or semicolon-separated, from: ${SANITIZER_KNOWN}")

if(CMAKE_BUILD_TYPE STREQUAL "Sanitizer")
    message(STATUS "Sanitizer Mode")
    set(SANITIZER_TYPE "ASAN" CACHE STRING "Choose the sanitizer to use.")
    set_property(CACHE SANITIZER_TYPE PROPERTY STRINGS
            "TSAN" "ASAN" "LSAN" "MSAN" "UBSAN")
else()
    unset(SANITIZER_TYPE CACHE)
endif()

# ENABLE_SANITIZERS wins; SANITIZER_TYPE is the pre-#1380 spelling and only
# applies to the dedicated build type.
set(sanitizer_requested "${ENABLE_SANITIZERS}")
if(NOT sanitizer_requested AND CMAKE_BUILD_TYPE STREQUAL "Sanitizer")
    set(sanitizer_requested "${SANITIZER_TYPE}")
endif()

set(sanitizer_list "")
string(REPLACE "," ";" sanitizer_requested "${sanitizer_requested}")
foreach(sanitizer IN LISTS sanitizer_requested)
    string(TOLOWER "${sanitizer}" sanitizer)
    if(DEFINED SANITIZER_ALIAS_${sanitizer})
        set(sanitizer "${SANITIZER_ALIAS_${sanitizer}}")
    endif()
    if(NOT sanitizer IN_LIST SANITIZER_KNOWN)
        message(FATAL_ERROR
                "Unknown sanitizer '${sanitizer}'; expected one of: ${SANITIZER_KNOWN}")
    endif()
    list(APPEND sanitizer_list "${sanitizer}")
endforeach()
list(REMOVE_DUPLICATES sanitizer_list)

if(CMAKE_BUILD_TYPE STREQUAL "Sanitizer")
    # The build type now carries only optimization and debug info; MSan keeps
    # the higher level it always had, to bound origin-tracking cost.
    set(sanitizer_opt "-g -O1")
    if("memory" IN_LIST sanitizer_list)
        set(sanitizer_opt "-g -O2")
    endif()

    set(CMAKE_C_FLAGS_SANITIZER "${sanitizer_opt}" CACHE
            STRING "C flags for sanitizer." FORCE)
    set(CMAKE_CXX_FLAGS_SANITIZER "${sanitizer_opt}" CACHE
            STRING "C++ flags for sanitizer." FORCE)
else()
    unset(CMAKE_C_FLAGS_SANITIZER CACHE)
    unset(CMAKE_CXX_FLAGS_SANITIZER CACHE)
endif()

if(sanitizer_list)
    set(sanitizer_flags "")
    foreach(sanitizer IN LISTS sanitizer_list)
        string(APPEND sanitizer_flags " ${SANITIZER_FLAGS_${sanitizer}}")
    endforeach()

    message(STATUS "Sanitizers enabled: ${sanitizer_list}")

    separate_arguments(sanitizer_flag_list UNIX_COMMAND "${sanitizer_flags}")
    add_compile_options(${sanitizer_flag_list})
    add_link_options(${sanitizer_flag_list})
endif()

# BuildStatic keys off this to skip -static, which drops the dynamic section the
# sanitizer runtimes need.
set(ESBMC_SANITIZERS_ENABLED "${sanitizer_list}")
