# Module to set flags for sanitizers

# Inspired on http://www.stablecoder.ca/2018/02/01/analyzer-build-types.html

if(CMAKE_BUILD_TYPE STREQUAL "Sanitizer")
    message(STATUS "Sanitizer Mode")
    set(SANITIZER_TYPE "ASAN" CACHE
            STRING "Choose the sanitizer to use.")

    set_property(CACHE SANITIZER_TYPE PROPERTY STRINGS
            "TSAN" "ASAN" "LSAN" "MSAN" "UBSAN")

    # ThreadSanitizer
    set(TSAN_FLAGS "-fsanitize=thread -g -O1")
    # AddressSanitizer.  -fsanitize-recover=address makes findings recoverable
    # so that, combined with halt_on_error=0, the process continues past a
    # report instead of aborting.  This is required for the build-time c2goto
    # invocations to survive bundled-clang findings (see sanitizers.yml) and
    # lets a single esbmc run surface more than one runtime finding.
    set(ASAN_FLAGS "-fsanitize=address -fsanitize-recover=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g -O1")
    # LeakSanitizer
    set(LSAN_FLAGS "-fsanitize=leak -fno-omit-frame-pointer -g -O1")
    # MemorySanitizer
    set(MSAN_FLAGS "-fsanitize=memory -fno-optimize-sibling-calls -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer -g -O2")
    # UndefinedBehaviour
    set(UBSAN_FLAGS "-fsanitize=undefined -fno-sanitize=vptr")

    set(CMAKE_C_FLAGS_SANITIZER
            "${${SANITIZER_TYPE}_FLAGS}" CACHE
            STRING "C flags for sanitizer." FORCE)
    set(CMAKE_CXX_FLAGS_SANITIZER
            "${${SANITIZER_TYPE}_FLAGS}"
            CACHE STRING "C++ flags for sanitizer." FORCE)
else()
    unset(SANITIZER_TYPE CACHE)
    unset(CMAKE_C_FLAGS_SANITIZER CACHE)
    unset(CMAKE_CXX_FLAGS_SANITIZER CACHE)
endif()
