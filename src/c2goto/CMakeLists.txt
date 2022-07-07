
add_subdirectory (headers)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/c2goto-hdr)
file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/c2goto-hdr/libc.h
           ${CMAKE_CURRENT_BINARY_DIR}/c2goto-hdr/libm.h)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/c2goto-hdr/headers)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/c2goto-hdr/headers/libc_hdr.h
           "#include \"${CMAKE_CURRENT_BINARY_DIR}/headers/libc_hdr.h\"")

add_executable(c2goto c2goto.cpp cprover_blank_library.cpp cprover_libc_sources.cpp)
target_include_directories(c2goto
    PRIVATE ${Boost_INCLUDE_DIRS}
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/c2goto-hdr
    )
target_link_libraries(c2goto 
  clangcfrontend gotoprograms langapi util_esbmc irep2 c2gotoheaders
  filesystem ${Boost_LIBRARIES})
set_target_properties(c2goto PROPERTIES
  PRIVATE_HEADER "${CMAKE_CURRENT_BINARY_DIR}/c2goto-hdr/libc.h;${CMAKE_CURRENT_BINARY_DIR}/c2goto-hdr/libm.h;${CMAKE_CURRENT_BINARY_DIR}/c2goto-hdr/headers/libc_hdr.h")


if(DEFINED C2GOTO_SYSROOT)
  set(multiarch_flags --sysroot "${C2GOTO_SYSROOT}")
endif()

if (ESBMC_BUNDLE_LIBC)
  set (clib32_config clib32.c --32 --fixedbv -D__ESBMC_FIXEDBV)
  set (clib64_config clib64.c --64 --fixedbv -D__ESBMC_FIXEDBV)
  set (clib32f_config clib32_fp.c --32 --floatbv)
  set (clib64f_config clib64_fp.c --64 --floatbv)
  set (inputs clib32_config
              clib64_config
              clib32f_config
              clib64f_config
      )
endif()

function(mangle_clib output)
  set(result "")
   file(GLOB c2goto_library_files CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/library/*.c")
  file(GLOB c2goto_libm_files CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/library/libm/*.c")
  file(GLOB_RECURSE c2goto_header_files CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/headers/*.h")
  foreach(in_f ${ARGN})
    set (in_f "${${in_f}}")
    list(GET in_f 0 in_file)
    list(SUBLIST in_f 1 -1 in_flags)
    file(RELATIVE_PATH out_file ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${in_file})
    string(REGEX REPLACE .c .goto out_goto "${in_file}")
    string(REGEX REPLACE .c "" barename "${in_file}")
    set(out_goto "${CMAKE_CURRENT_BINARY_DIR}/${out_goto}")
    set(out_file "${CMAKE_CURRENT_BINARY_DIR}/${out_file}")

    if(ENABLE_LIBM)
      set(inputs_c ${c2goto_library_files} ${c2goto_libm_files})
    else()
      set(inputs_c ${c2goto_library_files})
    endif()
    set(CMD c2goto ${multiarch_flags} ${OS_C2GOTO_FLAGS}
                   ${inputs_c} ${in_flags} --output ${out_goto})
    add_custom_command(OUTPUT ${out_goto}
      COMMAND ${CMD}
      DEPENDS c2goto ${c2goto_library_files} ${c2goto_libm_files} ${c2goto_header_files}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating libc model ${out_goto}"
      VERBATIM
      )

    add_custom_command(OUTPUT ${out_file}
      COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/flail.py -o ${out_file} ${out_goto}
      DEPENDS ${out_goto} ${CMAKE_SOURCE_DIR}/scripts/flail.py
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Converting libc model ${out_goto} to data"
      VERBATIM
      )

    list(APPEND result ${out_file})
  endforeach()
  mangle(${CMAKE_CURRENT_BINARY_DIR}/libc.c ${CMAKE_CURRENT_SOURCE_DIR}/library
         WILDCARD *.c SINGLE PREFIX esbmc_libc_ MACRO ESBMC_FLAIL
         ACC_HEADERS_INTO ${CMAKE_CURRENT_BINARY_DIR}/libc.h)
  mangle(${CMAKE_CURRENT_BINARY_DIR}/libm.c ${CMAKE_CURRENT_SOURCE_DIR}/library/libm
         WILDCARD *.c SINGLE PREFIX esbmc_libm_ MACRO ESBMC_FLAIL
         ACC_HEADERS_INTO ${CMAKE_CURRENT_BINARY_DIR}/libm.h)
  list(APPEND result ${CMAKE_CURRENT_BINARY_DIR}/libc.c
                     ${CMAKE_CURRENT_BINARY_DIR}/libm.c)
  set(${output} "${result}" PARENT_SCOPE)
endfunction()

message(STATUS "Multiarch flags for c2goto: ${multiarch_flags}. You may want to override C2GOTO_SYSROOT")
mangle_clib(mangled_files ${inputs})
add_library(clibs ${mangled_files} cprover_library.cpp cprover_libc_sources.cpp)
target_include_directories(clibs
    PRIVATE ${Boost_INCLUDE_DIRS}
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(clibs c2gotoheaders fmt::fmt filesystem)
set_target_properties(clibs PROPERTIES PRIVATE_HEADER "${CMAKE_CURRENT_BINARY_DIR}/libc.h;${CMAKE_CURRENT_BINARY_DIR}/libm.h")
