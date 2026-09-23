add_subdirectory(headers)

# c2goto reads the model sources it compiles out of the same VFS that ESBMC
# serves them from, so the goto binary records /esbmc-vfs paths rather than this
# machine's build tree. Hence the generated bundle lists on its include path: it
# needs the real ones.
add_executable(c2goto c2goto.cpp cprover_blank_library.cpp cprover_libc_sources.cpp)
target_include_directories(c2goto
    PRIVATE ${Boost_INCLUDE_DIRS}
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    )
target_link_libraries(c2goto
  clangcfrontend clangcppfrontend langapi c2gotoheaders
  filesystem ${Boost_LIBRARIES} ${OS_INCLUDE_LIBS})


if(C2GOTO_SYSROOT)
  set(multiarch_flags --sysroot "${C2GOTO_SYSROOT}")
endif()

function(foreach_in_cross_product_rec fn acc_i acc_v)
  list(LENGTH ARGN n)
  if(n EQUAL 0)
    # quotes around ${acc_v} are necessary since CMake does not handle empty
    # 'list' elements, sigh...
    cmake_language(CALL ${fn} "${acc_i}" "${acc_v}")
  else()
    list(GET ARGN 0 x)
    if(n GREATER 1)
      list(SUBLIST ARGN 1 -1 xs)
    else()
      list(SUBLIST ARGN 0 0 xs)
    endif()
    set(n 0)
    foreach(item IN LISTS ${x})
      set(ai ${acc_i})
      list(APPEND ai "${n}")
      set(av "${acc_v}")
      list(APPEND av "${item}")
      math(EXPR n "${n}+1")
      foreach_in_cross_product_rec(${fn} "${ai}" "${av}" ${xs})
    endforeach()
  endif()
endfunction()

function(foreach_in_cross_product fn)
  foreach_in_cross_product_rec(${fn} "" "" ${ARGN})
endfunction()

if(ESBMC_BUNDLE_LIBC)

  set(archs --64)
  if(ENABLE_BUNDLE_LIBC_32BIT)
    set(archs --32 --64)
  endif()
  set(floats "--fixedbv -D__ESBMC_FIXEDBV" --floatbv)
  # for some reason CMake does not differentiate between [""], "" and [] so we
  # have to 'represent' the empty string by a non-empty string in order to get
  # the non-cheri combinations, sigh...
  set(cherimodes " ")
  if(ESBMC_CHERI_CLANG)
    if(ESBMC_CHERI_HYBRID_SYSROOT)
      list(APPEND cherimodes "--cheri hybrid --sysroot ${ESBMC_CHERI_HYBRID_SYSROOT}")
    endif()
    if(ESBMC_CHERI_PURECAP_SYSROOT)
      list(APPEND cherimodes "--cheri purecap --sysroot ${ESBMC_CHERI_PURECAP_SYSROOT}")
    endif()
  endif()

  function(clib_join_argv idcs tuple)
    #message(STATUS "test: idcs: ${idcs}, tuple: ${tuple}")
    list(JOIN idcs "_" idcs)
    list(PREPEND tuple clib_${idcs}.c)
    list(JOIN tuple " " clib_config_${idcs})
    list(APPEND clib_configs clib_config_${idcs})
    message(STATUS "test: ${clib_config_${idcs}}")
  endfunction()

  foreach_in_cross_product(clib_join_argv cherimodes floats archs)

  set(clib64_config clib64.c --64 --fixedbv -D__ESBMC_FIXEDBV)
  set(clib64f_config clib64_fp.c --64 --floatbv)
  set(inputs clib64_config clib64f_config)
  if(ENABLE_BUNDLE_LIBC_32BIT)
    set(clib32_config clib32.c --32 --fixedbv -D__ESBMC_FIXEDBV)
    set(clib32f_config clib32_fp.c --32 --floatbv)
    set(inputs clib32_config
      clib64_config
      clib32f_config
      clib64f_config)
  endif()

  if(ESBMC_CHERI_HYBRID_SYSROOT)
    set(clib32_cherih_config clib32_cherih.c --32 --fixedbv -D__ESBMC_FIXEDBV --cheri hybrid --sysroot "${ESBMC_CHERI_HYBRID_SYSROOT}")
    set(clib64_cherih_config clib64_cherih.c --64 --fixedbv -D__ESBMC_FIXEDBV --cheri hybrid --sysroot "${ESBMC_CHERI_HYBRID_SYSROOT}")
    set(clib32f_cherih_config clib32_fp_cherih.c --32 --floatbv --cheri hybrid --sysroot "${ESBMC_CHERI_HYBRID_SYSROOT}")
    set(clib64f_cherih_config clib64_fp_cherih.c --64 --floatbv --cheri hybrid --sysroot "${ESBMC_CHERI_HYBRID_SYSROOT}")
    list(APPEND inputs
              #clib32_cherih_config
              clib64_cherih_config
              #clib32f_cherih_config
              clib64f_cherih_config)
  endif()

  if(ESBMC_CHERI_PURECAP_SYSROOT)
    set(clib32_cherip_config clib32_cherip.c --32 --fixedbv -D__ESBMC_FIXEDBV --cheri purecap --sysroot "${ESBMC_CHERI_PURECAP_SYSROOT}")
    set(clib64_cherip_config clib64_cherip.c --64 --fixedbv -D__ESBMC_FIXEDBV --cheri purecap --sysroot "${ESBMC_CHERI_PURECAP_SYSROOT}")
    set(clib32f_cherip_config clib32_fp_cherip.c --32 --floatbv --cheri purecap --sysroot "${ESBMC_CHERI_PURECAP_SYSROOT}")
    set(clib64f_cherip_config clib64_fp_cherip.c --64 --floatbv --cheri purecap --sysroot "${ESBMC_CHERI_PURECAP_SYSROOT}")
    list(APPEND inputs
              #clib32_cherip_config
              clib64_cherip_config
              #clib32f_cherip_config
              clib64f_cherip_config)
  endif()
endif()

function(mangle_clib output)
  set(result "")
  # Extract a list of C's operational models
  file(GLOB c2goto_library_files CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/library/*.c")
  # Extract a list of C's math operational models
  file(GLOB c2goto_libm_files CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/library/libm/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/library/libm/musl/*.c")
  file(GLOB_RECURSE c2goto_header_files CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/headers/*.h")

  # Extract C models for Python
  if(ENABLE_PYTHON_FRONTEND)
    file(GLOB_RECURSE c2goto_python_files CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/library/python/*.c")
  else()
    set(c2goto_python_files "")
  endif()

  # Extract C models for Solidity
  if(ENABLE_SOLIDITY_FRONTEND)
    file(GLOB_RECURSE c2goto_solidity_files CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/library/solidity/*.c")
  else()
    set(c2goto_solidity_files "")
  endif()

  # Extract C++ operational models (esbmc/esbmc#5298). These are compiled by
  # the C++ frontend inside c2goto and serialized into the embedded clib.
  file(GLOB_RECURSE c2goto_cpp_files CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/library/cpp/*.cpp")

  # For each config, generate clib*.c and clib*.goto file in the current binary directory
  foreach(in_f ${ARGN})
    set(in_f "${${in_f}}")
    list(GET in_f 0 out_name)
    list(SUBLIST in_f 1 -1 in_flags)
    file(RELATIVE_PATH out_file ${CMAKE_CURRENT_SOURCE_DIR}
                                ${CMAKE_CURRENT_SOURCE_DIR}/${out_name})
    string(REGEX REPLACE \\.c$ .goto out_goto "${out_name}")
    set(out_goto "${CMAKE_CURRENT_BINARY_DIR}/${out_goto}")
    set(out_file "${CMAKE_CURRENT_BINARY_DIR}/${out_file}")

    set(inputs_c ${c2goto_library_files})

    if(ENABLE_LIBM)
      list(APPEND inputs_c ${c2goto_libm_files})
    endif()

    # C++ operational models (esbmc/esbmc#5298): compiled by the C++ frontend
    # registered in c2goto and serialized into the clib alongside the C models.
    list(APPEND inputs_c ${c2goto_cpp_files})

    # Python (py64) and Solidity (sol64) models are built as their own goto
    # binaries, NOT mixed into clib, so a run of one language does not
    # deserialise the others' models (#6831).

    if(ESBMC_CHERI_CLANG)
      list(APPEND in_flags -I "${cheri_compressed_cap_SOURCE_DIR}")
    endif()

    model_vfs_paths(inputs_vfs ${inputs_c})
    set(run_c2goto c2goto ${multiarch_flags} ${OS_C2GOTO_FLAGS}
                   ${inputs_vfs} ${in_flags} --output ${out_goto})

    # Convert a list of C operational models (e.g. *.c) into a single goto binary file (e.g. clib32.goto) at CMake build time
    add_custom_command(OUTPUT ${out_goto}
      COMMAND ${run_c2goto}
      DEPENDS c2goto ${c2goto_library_files} ${c2goto_libm_files} ${c2goto_header_files} ${c2goto_cpp_files}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating libc model ${out_goto}"
      VERBATIM
      )

    # Convert the content of goto binary file (e.g. clib32.goto) into a C source file (e.g. clib32.c) containing the content of the binary
    add_custom_command(OUTPUT ${out_file}
      COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/flail.py -o ${out_file} ${out_goto}
      DEPENDS ${out_goto} ${CMAKE_SOURCE_DIR}/scripts/flail.py
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Converting libc model ${out_goto} to data"
      VERBATIM
      )

    list(APPEND result ${out_file})
  endforeach()

  # `mangle` function defined in src/scripts/cmake/Utils.cmake
  mangle(${CMAKE_CURRENT_BINARY_DIR}/libc.c  # an aggregate of C operational models
         ${CMAKE_CURRENT_SOURCE_DIR}/library # the directory containing all C operational models
         WILDCARD *.c
         SINGLE
         PREFIX esbmc_libc_
         MACRO ESBMC_FLAIL
         ACC_HEADERS_INTO ${CMAKE_CURRENT_BINARY_DIR}/libc.h)
  mangle(${CMAKE_CURRENT_BINARY_DIR}/libm.c       # an aggregate of C math operational models
         ${CMAKE_CURRENT_SOURCE_DIR}/library/libm # the directory C math operational models
         WILDCARD *.c
         SINGLE
         PREFIX esbmc_libm_
         MACRO ESBMC_FLAIL
         ACC_HEADERS_INTO ${CMAKE_CURRENT_BINARY_DIR}/libm.h)
  # Kept out of libm.h: add_bundled_library_sources() parses that list, and musl
  # is compiled into the clib but has never been part of the parsed fallback.
  mangle(${CMAKE_CURRENT_BINARY_DIR}/libmusl.c
         ${CMAKE_CURRENT_SOURCE_DIR}/library/libm/musl
         WILDCARD *.c
         SINGLE
         PREFIX esbmc_libmusl_
         MACRO ESBMC_FLAIL
         ACC_HEADERS_INTO ${CMAKE_CURRENT_BINARY_DIR}/libmusl.h)
  # musl's sources include each other by relative path, so its private headers
  # have to sit beside them in the VFS. Separate list because flail.py drops the
  # extension when it derives a symbol name, so exp2f_data.{c,h} would collide.
  mangle(${CMAKE_CURRENT_BINARY_DIR}/libmuslhdr.c
         ${CMAKE_CURRENT_SOURCE_DIR}/library/libm/musl
         WILDCARD *.h
         SINGLE
         PREFIX esbmc_libmuslhdr_
         MACRO ESBMC_FLAIL
         ACC_HEADERS_INTO ${CMAKE_CURRENT_BINARY_DIR}/libmuslhdr.h)
  # RECURSIVE, like the file(GLOB_RECURSE) that hands c2goto the same set: a
  # source it compiles by VFS path but that no mangle registered fails the build.
  mangle(${CMAKE_CURRENT_BINARY_DIR}/libcpp.c
         ${CMAKE_CURRENT_SOURCE_DIR}/library/cpp
         WILDCARD *.cpp
         RECURSIVE 1
         SINGLE
         PREFIX esbmc_libcpp_
         MACRO ESBMC_FLAIL
         ACC_HEADERS_INTO ${CMAKE_CURRENT_BINARY_DIR}/libcpp.h)

  if(ENABLE_PYTHON_FRONTEND)
    mangle(${CMAKE_CURRENT_BINARY_DIR}/libpython.c
           ${CMAKE_CURRENT_SOURCE_DIR}/library/python
           WILDCARD *.c
           RECURSIVE 1
           SINGLE
           PREFIX esbmc_python_
           MACRO ESBMC_FLAIL
           ACC_HEADERS_INTO ${CMAKE_CURRENT_BINARY_DIR}/libpython.h)
    # Separate list for the same reason as libmuslhdr: the models include
    # python_types.h by relative path, and flail.py keys symbols on the stem.
    mangle(${CMAKE_CURRENT_BINARY_DIR}/libpythonhdr.c
           ${CMAKE_CURRENT_SOURCE_DIR}/library/python
           WILDCARD *.h
           RECURSIVE 1
           SINGLE
           PREFIX esbmc_pythonhdr_
           MACRO ESBMC_FLAIL
           ACC_HEADERS_INTO ${CMAKE_CURRENT_BINARY_DIR}/libpythonhdr.h)

    # Build the Python operational models as their own goto binaries, the way
    # sol64 is built, so a C or C++ run does not deserialise them (#6831). Two
    # float encodings, matching the clib64/clib64_fp pair the runtime selects
    # between; the Python frontend is 64-bit only, so there is no 32-bit pair.
    set(py64_configs py64 py64_fp)
    set(py64_flags_py64 --fixedbv -D__ESBMC_FIXEDBV)
    set(py64_flags_py64_fp --floatbv)
    foreach(py_cfg ${py64_configs})
      set(py_goto "${CMAKE_CURRENT_BINARY_DIR}/${py_cfg}.goto")
      set(py_c "${CMAKE_CURRENT_BINARY_DIR}/${py_cfg}.c")
      add_custom_command(OUTPUT ${py_goto}
        COMMAND c2goto ${multiarch_flags} ${OS_C2GOTO_FLAGS}
                ${c2goto_python_files} --64 ${py64_flags_${py_cfg}}
                --output ${py_goto}
        DEPENDS c2goto ${c2goto_python_files} ${c2goto_header_files}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating Python model ${py_goto}"
        VERBATIM
      )
      add_custom_command(OUTPUT ${py_c}
        COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/flail.py
                -o ${py_c} ${py_goto}
        DEPENDS ${py_goto} ${CMAKE_SOURCE_DIR}/scripts/flail.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Converting Python model ${py_goto} to data"
        VERBATIM
      )
      list(APPEND result ${py_c})
    endforeach()

    list(APPEND result
      ${CMAKE_CURRENT_BINARY_DIR}/libpython.c
      ${CMAKE_CURRENT_BINARY_DIR}/libpython.h
      ${CMAKE_CURRENT_BINARY_DIR}/libpythonhdr.c
      ${CMAKE_CURRENT_BINARY_DIR}/libpythonhdr.h)
  endif()

  if(ENABLE_SOLIDITY_FRONTEND)
    mangle(${CMAKE_CURRENT_BINARY_DIR}/libsolidity.c
           ${CMAKE_CURRENT_SOURCE_DIR}/library/solidity
           WILDCARD *.c
           RECURSIVE 1
           SINGLE
           PREFIX esbmc_solidity_
           MACRO ESBMC_FLAIL
           ACC_HEADERS_INTO ${CMAKE_CURRENT_BINARY_DIR}/libsolidity.h)
    # Separate list, as for libmuslhdr: solidity_types.h shares no stem with a
    # .c model today, but keeping headers apart makes that a non-issue.
    mangle(${CMAKE_CURRENT_BINARY_DIR}/libsolidityhdr.c
           ${CMAKE_CURRENT_SOURCE_DIR}/library/solidity
           WILDCARD *.h
           RECURSIVE 1
           SINGLE
           PREFIX esbmc_solidityhdr_
           MACRO ESBMC_FLAIL
           ACC_HEADERS_INTO ${CMAKE_CURRENT_BINARY_DIR}/libsolidityhdr.h)

    # Build a separate goto binary for Solidity operational models (sol64.goto)
    # This is loaded at runtime instead of the full clib64.goto for performance.
    set(sol64_goto "${CMAKE_CURRENT_BINARY_DIR}/sol64.goto")
    set(sol64_c "${CMAKE_CURRENT_BINARY_DIR}/sol64.c")
    if(APPLE OR (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64"))
      # Clang's ARM backend does not support _BitInt > 128, so c2goto cannot
      # compile solidity_types.h on ARM64 (macOS or Linux). Emit an empty stub
      # so the link succeeds; runtime Solidity verification is unavailable on ARM64.
      file(WRITE ${sol64_c}
        "/* Auto-generated stub: sol64 unavailable on macOS (_BitInt>128). */\n"
        "#include <stdint.h>\n"
        "const uint8_t sol64_buf[1] = {0};\n"
        "const unsigned int sol64_buf_size = 0;\n")
    else()
      # Amalgamate all solidity .c files into one TU. Passing the originals
      # separately makes c2goto invoke clang::ASTImporter to merge them, which
      # on LLVM 16 (Windows) cannot import _BitInt(N)-using decls and emits
      # "UnsupportedConstruct" until the whole step aborts. A single TU avoids
      # the importer entirely.
      set(sol64_unified_c "${CMAKE_CURRENT_BINARY_DIR}/sol64_unified.c")
      set(sol64_unified_body
        "/* Auto-generated by CMake: amalgamation of solidity *.c models. */\n")
      # VFS paths, so the models' symbols carry no build-tree path. The
      # amalgamation itself contributes no code, so its own path never appears.
      model_vfs_paths(sol64_vfs ${c2goto_solidity_files})
      foreach(_f ${sol64_vfs})
        string(APPEND sol64_unified_body "#include \"${_f}\"\n")
      endforeach()
      file(WRITE ${sol64_unified_c} "${sol64_unified_body}")

      # No --fixedbv here: the contract under verification, the default clib and
      # the solver encoding all run floats as IEEE, so building the models with
      # the legacy fixed-point fallback gives the same C type two incompatible
      # representations either side of the model boundary (#6898).
      add_custom_command(OUTPUT ${sol64_goto}
        COMMAND c2goto ${multiarch_flags} ${OS_C2GOTO_FLAGS}
                ${sol64_unified_c} --64
                --output ${sol64_goto}
        DEPENDS c2goto ${c2goto_solidity_files} ${c2goto_header_files}
                ${sol64_unified_c}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating Solidity model ${sol64_goto}"
        VERBATIM
      )
      add_custom_command(OUTPUT ${sol64_c}
        COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/flail.py
                -o ${sol64_c} ${sol64_goto}
        DEPENDS ${sol64_goto} ${CMAKE_SOURCE_DIR}/scripts/flail.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Converting Solidity model ${sol64_goto} to data"
        VERBATIM
      )
    endif()

    list(APPEND result
      ${CMAKE_CURRENT_BINARY_DIR}/libsolidity.c
      ${CMAKE_CURRENT_BINARY_DIR}/libsolidity.h
      ${CMAKE_CURRENT_BINARY_DIR}/libsolidityhdr.c
      ${CMAKE_CURRENT_BINARY_DIR}/libsolidityhdr.h
      ${sol64_c})
  endif()


  list(APPEND result ${CMAKE_CURRENT_BINARY_DIR}/libc.c
                     ${CMAKE_CURRENT_BINARY_DIR}/libm.c
                     ${CMAKE_CURRENT_BINARY_DIR}/libmusl.c
                     ${CMAKE_CURRENT_BINARY_DIR}/libmuslhdr.c
                     ${CMAKE_CURRENT_BINARY_DIR}/libcpp.c)
  set(${output} "${result}" PARENT_SCOPE)
endfunction()

# Absolute model-source paths -> the VFS paths c2goto is handed instead, so no
# build-tree path is baked into the goto binary. Mirrors the prefixes
# register_bundled_libc() registers them under.
if(WIN32)
  # Must track clang_vfs_path(): the registry keys carry the drive letter there,
  # so a bare "/esbmc-vfs/..." would miss every lookup.
  set(vfs_root "C:/esbmc-vfs")
else()
  set(vfs_root "/esbmc-vfs")
endif()

function(model_vfs_paths output)
  set(result "")
  foreach(f ${ARGN})
    file(RELATIVE_PATH rel "${CMAKE_CURRENT_SOURCE_DIR}/library" "${f}")
    # An input outside library/ escapes the prefix the mangles register, so the
    # VFS path would resolve to nothing and only surface as c2goto failing to
    # open it.
    if(rel MATCHES "^\\.\\.(/|$)")
      message(FATAL_ERROR "c2goto model source outside library/: ${f}")
    endif()
    list(APPEND result "${vfs_root}/libc/library/${rel}")
  endforeach()
  set(${output} "${result}" PARENT_SCOPE)
endfunction()

message(STATUS "Multiarch flags for c2goto: ${multiarch_flags}. You may want to override C2GOTO_SYSROOT")
mangle_clib(mangled_files ${inputs})

# The model sources themselves, so c2goto can serve them to clang out of the VFS
# rather than reading this machine's build tree. Data only, generated by mangle()
# without c2goto's involvement, so this is not a dependency cycle.
set(c2goto_model_data
  ${CMAKE_CURRENT_BINARY_DIR}/libc.c
  ${CMAKE_CURRENT_BINARY_DIR}/libm.c
  ${CMAKE_CURRENT_BINARY_DIR}/libmusl.c
  ${CMAKE_CURRENT_BINARY_DIR}/libmuslhdr.c
  ${CMAKE_CURRENT_BINARY_DIR}/libcpp.c)
if(ENABLE_PYTHON_FRONTEND)
  list(APPEND c2goto_model_data ${CMAKE_CURRENT_BINARY_DIR}/libpython.c
                                ${CMAKE_CURRENT_BINARY_DIR}/libpythonhdr.c)
endif()
if(ENABLE_SOLIDITY_FRONTEND)
  list(APPEND c2goto_model_data ${CMAKE_CURRENT_BINARY_DIR}/libsolidity.c
                                ${CMAKE_CURRENT_BINARY_DIR}/libsolidityhdr.c)
endif()
target_sources(c2goto PRIVATE ${c2goto_model_data})

# Auto-generate the Python operational-model allowlist from the model sources so
# new models register automatically (scripts/gen_python_c_models.py). The header
# is consumed by cprover_library.cpp under #ifdef ENABLE_PYTHON_FRONTEND.
if(ENABLE_PYTHON_FRONTEND)
  find_package(Python REQUIRED COMPONENTS Interpreter)
  file(GLOB_RECURSE python_model_sources CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/library/python/*.c")
  set(python_c_models_header ${CMAKE_CURRENT_BINARY_DIR}/python_c_models_generated.h)
  add_custom_command(OUTPUT ${python_c_models_header}
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/gen_python_c_models.py
            -o ${python_c_models_header} ${python_model_sources}
    DEPENDS ${CMAKE_SOURCE_DIR}/scripts/gen_python_c_models.py ${python_model_sources}
    COMMENT "Generating Python C-model allowlist ${python_c_models_header}"
    VERBATIM)
  add_custom_target(gen_python_c_models DEPENDS ${python_c_models_header})
endif()

add_library(clibs ${mangled_files} cprover_library.cpp cprover_libc_sources.cpp)
target_include_directories(clibs
    PRIVATE ${Boost_INCLUDE_DIRS}
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(clibs c2gotoheaders fmt::fmt filesystem)
if(ENABLE_PYTHON_FRONTEND)
  add_dependencies(clibs gen_python_c_models)
endif()

set(private_headers
  "${CMAKE_CURRENT_BINARY_DIR}/libc.h"
  "${CMAKE_CURRENT_BINARY_DIR}/libm.h"
  "${CMAKE_CURRENT_BINARY_DIR}/libmusl.h"
  "${CMAKE_CURRENT_BINARY_DIR}/libmuslhdr.h"
  "${CMAKE_CURRENT_BINARY_DIR}/libcpp.h"
)

if(ENABLE_PYTHON_FRONTEND)
  list(APPEND private_headers "${CMAKE_CURRENT_BINARY_DIR}/libpython.h"
                              "${CMAKE_CURRENT_BINARY_DIR}/libpythonhdr.h")
endif()

if(ENABLE_SOLIDITY_FRONTEND)
  list(APPEND private_headers "${CMAKE_CURRENT_BINARY_DIR}/libsolidity.h"
                              "${CMAKE_CURRENT_BINARY_DIR}/libsolidityhdr.h")
endif()

set_target_properties(clibs PROPERTIES PRIVATE_HEADER "${private_headers}")


