include_guard(GLOBAL)

if(ENABLE_COVERAGE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
        message(FATAL_ERROR
            "ENABLE_COVERAGE requires Clang or AppleClang (found "
            "${CMAKE_CXX_COMPILER_ID}); configure with -DENABLE_COVERAGE=Off "
            "or point CMAKE_CXX_COMPILER at clang++.")
    endif()

    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "Code coverage results with an optimized (non-Debug) build may be misleading")
    endif()

    find_program(LCOV_PATH lcov)
    if(NOT LCOV_PATH)
        message(WARNING "lcov not found — coverage reports will not be filtered (filtered LCOV report will not be produced)")
    endif()

    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate)

    # Instrument ESBMC's own sources only. Clang 18 also instruments every
    # libstdc++/LLVM/boost/fmt template instantiated from /usr headers: about
    # 80% of the functions in the esbmc binary, so most of each per-process
    # .profraw (24 MB each, ~5,500 of them per CI run) recorded code the
    # coverage report then strips out anyway. Skipping it here shrinks the raw
    # files without changing any line count in ESBMC's own sources.
    set(_esbmc_profile_list
        "[clang]\ndefault:allow\nsource:/usr/*=skip\nsource:${CMAKE_BINARY_DIR}/*=skip\n")
    # ccache hashes the command line but not the list's contents, so put a
    # content hash in the file name to invalidate cached objects on any edit.
    string(MD5 _esbmc_profile_list_hash "${_esbmc_profile_list}")
    set(_esbmc_profile_list_file
        "${CMAKE_BINARY_DIR}/coverage-profile-list-${_esbmc_profile_list_hash}.txt")
    file(CONFIGURE OUTPUT "${_esbmc_profile_list_file}"
         CONTENT "${_esbmc_profile_list}" @ONLY)
    add_compile_options("-fprofile-list=${_esbmc_profile_list_file}")
endif()
